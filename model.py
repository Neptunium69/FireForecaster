from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from utae import UTAE
from utae.utae import DownConvBlock
from alpaca.moc import Memory
from torch.utils.checkpoint import checkpoint


class FireForecast(nn.Module):
    """Encode history, optionally transform it with memory, then decode features."""

    def __init__(self, input_channels, memory=None, *, low_vram=False, gpu_cache_gb=5.0):
        super().__init__()
        self.encoder = FireHistoryEncoder(input_channels, low_vram=low_vram, gpu_cache_gb=gpu_cache_gb)
        self.memory = memory
        self.decoder = UNetDecoder()
        self.output=OutputModule()

    def forward(self, history, dates,burned=None):
        bottleneck, skips = self.encoder(history, dates)
        if self.memory is not None:
            # Keep recurrent memory arithmetic in float32.
            with torch.autocast(device_type=bottleneck.device.type, enabled=False):
                transformed = self.memory(bottleneck.float())
            bottleneck = transformed
        fmap=self.decoder(bottleneck, skips)
        return self.output(fmap,burned)

class FireHistoryEncoder(nn.Module):

    def __init__(self, input_channels, *, low_vram=False, gpu_cache_gb=5.0):
        super().__init__()
        self.low_vram = low_vram
        self.gpu_cache_bytes = int(gpu_cache_gb * 1024**3)
        utae = UTAE(
            input_dim=input_channels,
            encoder_widths=[32, 32, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            n_head=8,
            d_model=128,
            d_k=4,
            agg_mode="att_group",
            encoder=True,
        )
        self.in_conv = utae.in_conv
        self.down_blocks = utae.down_blocks
        self.temporal_encoder = utae.temporal_encoder
        self.temporal_aggregator = utae.temporal_aggregator
        self.pad_value = utae.pad_value

    def forward(self, history, dates):
        if self.low_vram:
            return self._forward_low_vram(history, dates)
        pad_mask = (history == self.pad_value).all(dim=(2, 3, 4))

        features = self.in_conv.smart_forward(history)
        feature_maps = [features]
        for block in self.down_blocks:
            down_block = cast(DownConvBlock, block)
            features = down_block.smart_forward(features)
            feature_maps.append(features)

        bottleneck, attention = self._attend_valid_history(
            feature_maps[-1], dates, pad_mask
        )
        skips = [
            self.temporal_aggregator(
                features, pad_mask=pad_mask, attn_mask=attention
            )
            for features in reversed(feature_maps[:-1])
        ]
        return bottleneck, skips

    def _encode_day(self, frame, keep_on_gpu=False):
        frame = frame.to(next(self.parameters()).device)
        padded = (frame == self.pad_value).all(dim=(1, 2, 3))[:, None, None, None]
        features = self.in_conv(frame).masked_fill(padded, self.pad_value)
        maps = [features if keep_on_gpu else features.cpu()]
        for block in self.down_blocks:
            features = block(features).masked_fill(padded, self.pad_value)
            maps.append(features if keep_on_gpu else features.cpu())
        return tuple(maps)

    @staticmethod
    def _weight_day(features, attention, valid):
        # Same att_group calculation, with only one day's spatial maps on GPU.
        features = features.to(attention.device)
        heads, batch, height, width = attention.shape
        weights = attention.reshape(heads * batch, 1, height, width)
        if features.shape[-2] > width:
            weights = F.interpolate(weights, size=features.shape[-2:],
                                    mode="bilinear", align_corners=False)
        else:
            weights = F.avg_pool2d(weights, kernel_size=width // features.shape[-2])
        weights = weights.reshape(heads, batch, 1, *features.shape[-2:])
        weights = weights * valid[None, :, None, None, None]
        groups = torch.stack(features.chunk(heads, dim=1))
        return torch.cat(list(groups * weights), dim=1)

    def _forward_low_vram(self, history, dates):
        device = next(self.parameters()).device
        pad_mask = (history == self.pad_value).all(dim=(2, 3, 4)).to(device)
        maps = []
        cached_bytes = 0
        frame_bytes = 0
        for day in range(history.shape[1]):
            frame = history[:, day]
            if frame.shape[0] == 1:
                # Keep batch stride fixed as history grows, without copying data.
                frame = frame[0].unsqueeze(0)
            # Keep a bounded feature cache on GPU; spill longer histories to CPU.
            keep_on_gpu = frame_bytes > 0 and cached_bytes + frame_bytes <= self.gpu_cache_bytes
            # Retain the first four days; recompute later CNN activations during backward.
            if self.training and torch.is_grad_enabled() and day >= 4:
                # Checkpoint only pure CNN work, never stateful memory or attention.
                encoded = checkpoint(self._encode_day, frame, keep_on_gpu, use_reentrant=False)
            else:
                encoded = self._encode_day(frame, keep_on_gpu)
            frame_bytes = sum(level.numel() * level.element_size() for level in encoded)
            if keep_on_gpu:
                cached_bytes += frame_bytes
            maps.append(encoded)
        deepest = torch.stack([m[-1].to(device) for m in maps], dim=1)
        # Run attention once: it contains dropout and BatchNorm running statistics.
        bottleneck, attention = self._attend_valid_history(deepest, dates.to(device), pad_mask)
        skips = []
        for level in (2, 1, 0):
            total = None
            for day, encoded in enumerate(maps):
                args = (encoded[level], attention[:, :, day], ~pad_mask[:, day])
                if self.training and torch.is_grad_enabled():
                    weighted = checkpoint(self._weight_day, *args, use_reentrant=False)
                else:
                    weighted = self._weight_day(*args)
                total = weighted if total is None else total + weighted
            skips.append(total)
        return bottleneck, skips

    def _attend_valid_history(self, features, dates, pad_mask):
        if not pad_mask.any():
            return self.temporal_encoder(features, batch_positions=dates)
        counts = (~pad_mask).sum(dim=1)
        outputs = [None] * features.shape[0]
        attentions = [None] * features.shape[0]
        for length in counts.unique().tolist():
            indices = (counts == length).nonzero(as_tuple=True)[0].tolist()
            group_features = torch.stack([
                features[index, ~pad_mask[index]] for index in indices
            ])
            group_dates = torch.stack([
                dates[index, ~pad_mask[index]] for index in indices
            ])
            encoded, attention = self.temporal_encoder(
                group_features, batch_positions=group_dates
            )
            for offset, index in enumerate(indices):
                outputs[index] = encoded[offset:offset + 1]
                restored = attention.new_zeros(
                    attention.shape[0], 1, features.shape[1],
                    features.shape[-2], features.shape[-1],
                )
                valid_indices = (~pad_mask[index]).nonzero(as_tuple=True)[0]
                attentions[index] = restored.index_copy(
                    2, valid_indices, attention[:, offset:offset + 1]
                )
        return torch.cat(outputs, dim=0), torch.cat(attentions, dim=1)


class UNetDecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, features, skip):
        features = F.interpolate(
            features, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fusion(torch.cat([features, skip], dim=1))


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            UNetDecoderBlock(128, 64, 64),
            UNetDecoderBlock(64, 32, 32),
            UNetDecoderBlock(32, 32, 32),
        ])

    def forward(self, bottleneck, skips):
        features = bottleneck
        for block, skip in zip(self.blocks, skips):
            features = block(features, skip)
        return features


class AppendMemory(nn.Module):
    def __init__(self, feature_channels=128, memory_width=512):
        super().__init__()
        self.projection = nn.Linear(memory_width, feature_channels)

    def forward(self, inputs):
        features, memory = inputs
        height, width = features.shape[-2:]
        memory = self.projection(memory)[:, :, None, None]
        memory = memory.expand(-1, -1, height, width)

        return torch.cat([features, memory], dim=1)


class MemoryAdapter(nn.Module):
    def __init__(self, high, low, dims, size, device, shape,*, memory_key="encoder", steps=8, dt=0.005,):
        super().__init__()
        self.memory = Memory(
            high, low, dims, size, device, interval=1, shape=shape, name="memory"
        )
        self.memory_key = memory_key
        self.steps = steps
        self.dt = dt
        self.encoder_encoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=1),
        )
        self.encoder_encoder_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )
        self.memory_decoder = nn.Sequential(
            AppendMemory(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.to(device)

    def forward(self, bottleneck):
        encoded = self.encoder_encoder(bottleneck)
        input_slice = self.memory.input_slices[self.memory_key]
        output_slice = self.memory.output_slices[self.memory_key]
        for _ in range(self.steps):
            readout = self.memory.voltage[output_slice].unsqueeze(0)
            new_input = self.encoder_encoder_mlp(torch.cat([encoded, readout], dim=1))
            stimulus = torch.zeros_like(self.memory.voltage)
            stimulus[input_slice] = new_input[0].to(stimulus)
            self.memory.step(stimulus, dt=self.dt)
        readout = self.memory.voltage[output_slice].unsqueeze(0)
        return bottleneck + self.memory_decoder((bottleneck, readout))

class OutputModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.existing_head=nn.Conv2d(32,1,1)
        self.burned_head=nn.Conv2d(32,1,1)
        self.fire_head=nn.Conv2d(32,1,1)

        self.merger=nn.Sequential(
            nn.Conv2d(33,32,3,padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

    def forward(self,features, burned=None):
        logits=None
        if burned is None:
            logits=self.existing_head(features)
            burned = torch.sigmoid(logits.float())
        
        transformed=self.merger(torch.cat([features,burned],dim=1))
        logits_onfire=self.fire_head(transformed)
        logits_burnt=self.burned_head(transformed)

        onfire=torch.sigmoid(logits_onfire.float())
        burnt=torch.sigmoid(logits_burnt.float())

        new = (1 - burned) * burnt
        burned =burned + new

        return (onfire,burned,new)
