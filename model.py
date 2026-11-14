from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from utae import UTAE
from utae.utae import DownConvBlock
from alpaca.moc import Memory


class FireForecast(nn.Module):
    """Encode history, optionally transform it with memory, then decode features."""

    def __init__(self, input_channels, memory=None):
        super().__init__()
        self.encoder = FireHistoryEncoder(input_channels)
        self.memory = memory
        self.decoder = UNetDecoder()
        self.output=OutputModule()

    def forward(self, history, dates,burned=None):
        bottleneck, skips = self.encoder(history, dates)
        if self.memory is not None:
            transformed = self.memory(bottleneck)
            bottleneck = transformed
        fmap=self.decoder(bottleneck, skips)
        return self.output(fmap,burned)

class FireHistoryEncoder(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
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
            burned = torch.sigmoid(logits)
        
        transformed=self.merger(torch.cat([features,burned],dim=1))
        logits_onfire=self.fire_head(transformed)
        logits_burnt=self.burned_head(transformed)

        onfire=torch.sigmoid(logits_onfire)
        burnt=torch.sigmoid(logits_burnt)

        new = (1 - burned) * burnt
        burned =burned + new

        return (onfire,burned,new)
