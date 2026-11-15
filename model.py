import math
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

    def forward(self, history, dates, burned=None, *, spatial_cache=None, update_memory=True):
        bottleneck, skips = self.encoder(history, dates, spatial_cache=spatial_cache)
        bottleneck = self.apply_memory(bottleneck, history[:, -1, 27:29], update_memory=update_memory)
        return self.output(self.decoder(bottleneck, skips), burned)

    def apply_memory(self, bottleneck, observed_fire_maps=None, *, update_memory=True):
        if self.memory is None:
            return bottleneck
        # Keep recurrent memory arithmetic in float32.
        with torch.autocast(device_type=bottleneck.device.type, enabled=False):
            bottleneck = bottleneck.float()
            if not update_memory:
                return self.memory.read(bottleneck)
            if isinstance(self.memory, MemoryAdapter):
                return self.memory(bottleneck, observed_fire_maps)
            return self.memory(bottleneck)

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

    def forward(self, history, dates, *, spatial_cache=None):
        if spatial_cache is not None:
            spatial_cache._prepare(history)
            pad_mask = torch.stack(spatial_cache.padding[:history.shape[1]], dim=1)
            if self.low_vram:
                return self._aggregate_daily_maps(spatial_cache.daily_maps(history), dates, pad_mask)
            return self._aggregate_maps(spatial_cache.feature_maps(history), dates, pad_mask)
        if self.low_vram:
            return self._forward_low_vram(history, dates)
        pad_mask = (history == self.pad_value).all(dim=(2, 3, 4))

        features = self.in_conv.smart_forward(history)
        feature_maps = [features]
        for block in self.down_blocks:
            down_block = cast(DownConvBlock, block)
            features = down_block.smart_forward(features)
            feature_maps.append(features)

        return self._aggregate_maps(feature_maps, dates, pad_mask)

    def _aggregate_maps(self, feature_maps, dates, pad_mask):
        bottleneck, attention = self._attend_valid_history(
            feature_maps[-1], dates, pad_mask
        )
        skips = [
            self._aggregate_skip(features, pad_mask, attention)
            for features in reversed(feature_maps[:-1])
        ]
        return bottleneck, skips

    @staticmethod
    def _aggregate_skip(features, pad_mask, attention):
        # Group channels with a view instead of copying the entire history per head.
        heads, batch, length, height, width = attention.shape
        weights = attention.reshape(heads * batch, length, height, width)
        if features.shape[-2] > width:
            weights = F.interpolate(weights, size=features.shape[-2:], mode="bilinear", align_corners=False)
        else:
            weights = F.avg_pool2d(weights, kernel_size=width // features.shape[-2])
        weights = weights.reshape(heads, batch, length, *features.shape[-2:])
        if pad_mask is not None and pad_mask.any():
            weights = weights * (~pad_mask).float()[None, :, :, None, None]
        grouped = features.reshape(batch, length, heads, features.shape[2] // heads, *features.shape[-2:])
        return (grouped * weights.permute(1, 2, 0, 3, 4).unsqueeze(3)).sum(dim=1).flatten(1, 2)

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
        return self._aggregate_daily_maps(maps, dates, pad_mask)

    def _aggregate_daily_maps(self, maps, dates, pad_mask):
        device = next(self.parameters()).device
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



class _CollectFeatureGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, accumulated, anchor):
        ctx.accumulated = accumulated
        return features

    @staticmethod
    def backward(ctx, gradient):
        # Sum in float32 even when the forward used bfloat16.
        ctx.accumulated.add_(gradient.to(ctx.accumulated))
        return None, None, None


class RolloutSpatialCache:
    def __init__(self, encoder, batch_size=None, capacity=None, retain_activations=None):
        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.training = torch.is_grad_enabled()
        self.anchor = torch.zeros((), device=self.device, requires_grad=self.training)
        self.entries = []
        self.gpu_bytes = 0
        self.batch_size = batch_size or (1 if encoder.low_vram else 8)
        self.capacity = capacity
        self.banks = self.grad_banks = None
        self.padding = []
        self.retain_activations = not encoder.low_vram if retain_activations is None else retain_activations
        self.retained = []
        self.replay_start = 0

    def _prepare(self, history):
        self.autocast_enabled = torch.is_autocast_enabled(self.device.type)
        self.autocast_dtype = torch.get_autocast_dtype(self.device.type)
        retain_prefix = self.training and self.retain_activations and not self.entries
        for start in range(len(self.entries), history.shape[1], self.batch_size):
            stop = min(start + self.batch_size, history.shape[1])
            batch = history[:, start:stop].reshape(-1, *history.shape[2:]).detach()
            # Retain up to 32 observed frames; keep generated-frame replay batched.
            retain = retain_prefix and stop <= 32
            with torch.set_grad_enabled(retain):
                encoded = self.encoder._encode_day(batch, keep_on_gpu=True)
            if retain:
                self.retained.append((start, stop, encoded))
                self.replay_start = stop
            if self.capacity is not None and not self.encoder.low_vram and self.banks is None:
                self.banks = tuple(x.new_empty(history.shape[0], self.capacity, *x.shape[1:]) for x in encoded)
                if self.training:
                    self.grad_banks = tuple(torch.zeros_like(x, dtype=torch.float32 if x.element_size() < 4 else x.dtype) for x in self.banks)
            for offset, day in enumerate(range(start, stop)):
                frame = history[:, day].detach()
                maps = tuple(x.detach().reshape(history.shape[0], stop-start, *x.shape[1:])[:, offset] for x in encoded)
                byte_count = sum(x.numel() * (x.element_size() + (max(4, x.element_size()) if self.training else 0)) for x in maps)
                on_gpu = not self.encoder.low_vram or self.gpu_bytes + byte_count <= self.encoder.gpu_cache_bytes
                storage = self.device if on_gpu else torch.device("cpu")
                if self.banks is not None:
                    for bank, features in zip(self.banks, maps):
                        bank[:, day].copy_(features)
                    maps = tuple(bank[:, day] for bank in self.banks)
                    gradients = tuple(bank[:, day] for bank in self.grad_banks) if self.training else None
                else:
                    maps = tuple(x.to(storage) for x in maps)
                    gradients = tuple(torch.zeros_like(x, dtype=torch.float32 if x.element_size() < 4 else x.dtype) for x in maps) if self.training else None
                self.padding.append((frame == self.encoder.pad_value).all(dim=(1, 2, 3)).to(self.device))
                self.entries.append((frame, maps, gradients))
                if on_gpu:
                    self.gpu_bytes += byte_count

    def daily_maps(self, history):
        self._prepare(history)
        days = []
        for _, maps, gradients in self.entries[:history.shape[1]]:
            day = []
            for level, features in enumerate(maps):
                if self.training:
                    features = _CollectFeatureGradient.apply(features, gradients[level], self.anchor)
                day.append(features)
            days.append(tuple(day))
        return days

    def feature_maps(self, history):
        self._prepare(history)
        if self.banks is not None:
            length = history.shape[1]
            if self.training:
                return [_CollectFeatureGradient.apply(x[:, :length], g[:, :length], self.anchor) for x, g in zip(self.banks, self.grad_banks)]
            return [x[:, :length] for x in self.banks]
        days = self.daily_maps(history)
        return [torch.stack([day[level].to(self.device) for day in days], dim=1) for level in range(4)]

    def backward(self):
        for start, stop, encoded in self.retained:
            entries = self.entries[start:stop]
            gradients = tuple(torch.stack([entry[2][level].to(self.device) for entry in entries], dim=1).flatten(0, 1) for level in range(4))
            torch.autograd.backward(encoded, gradients)
        self.retained.clear()
        for start in range(self.replay_start, len(self.entries), self.batch_size):
            entries = self.entries[start:start+self.batch_size]
            frames = torch.cat([entry[0] for entry in entries], dim=0)
            with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled):
                encoded = self.encoder._encode_day(frames, keep_on_gpu=True)
            gradients = tuple(torch.cat([entry[2][level].to(self.device) for entry in entries], dim=0) for level in range(4))
            torch.autograd.backward(encoded, gradients)

    def clear(self):
        self.entries.clear()
        self.padding.clear()
        self.retained.clear()
        self.replay_start = 0
        self.banks = self.grad_banks = None
        self.gpu_bytes = 0


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
        memory = self.projection(memory)
        if memory.ndim == 2:
            memory = memory[:, :, None, None].expand(-1, -1, *features.shape[-2:])
        else:
            memory = memory.permute(0, 3, 1, 2)
            memory = F.interpolate(memory, size=features.shape[-2:], mode="bilinear", align_corners=False)

        return torch.cat([features, memory], dim=1)

class LegacyMemoryAdapter(nn.Module):
    """Use one shared SNN, with optional independent state for each map region."""

    def __init__(self, high, low, dims, size, device, shape,*, memory_key="encoder", steps=8, dt=0.005, input_gain=8.0, grid_size=None):
        super().__init__()
        if grid_size is not None:
            grid_size = tuple(grid_size)
        self.grid_size = grid_size
        self.memory = Memory(
            high, low, dims, size, device, interval=1, shape=shape, name="memory"
        )
        if grid_size is not None:
            state_shape = (math.prod(grid_size), size)
            self.memory.voltage = self.memory.voltage.new_zeros(state_shape)
            self.reset_memory()
        self.memory_key = memory_key
        self.steps = steps
        self.dt = dt
        self.input_gain = input_gain
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

    def reset_memory(self):
        """Start a new fire with zero voltage and spikes in every region."""
        memory = self.memory
        memory.voltage = torch.zeros_like(memory.voltage)
        memory.spikes = torch.zeros_like(memory.voltage)
        memory.y = {
            key: memory.voltage.new_zeros((*memory.voltage.shape[:-1], width))
            for key, width in memory.shape.items()
        }

    def detach_memory(self):
        """Keep the stored state while ending its autograd history."""
        memory = self.memory
        memory.voltage = memory.voltage.detach()
        memory.spikes = memory.spikes.detach()
        memory.y = {key: value.detach() for key, value in memory.y.items()}

    def _regions(self, bottleneck):
        if self.grid_size is None:
            return bottleneck
        rows, cols = self.grid_size
        height, width = bottleneck.shape[-2:]
        tile_h, tile_w = math.ceil(height / rows), math.ceil(width / cols)
        divisible_size = (rows * tile_h, cols * tile_w)
        if divisible_size != (height, width):
            bottleneck = F.interpolate(bottleneck, size=divisible_size, mode="bilinear", align_corners=False)
        channels = bottleneck.shape[1]
        tiles = bottleneck[0].reshape(channels, rows, tile_h, cols, tile_w)
        return tiles.permute(1, 3, 0, 2, 4).reshape(rows * cols, channels, tile_h, tile_w)

    def forward(self, bottleneck):
        encoded = self.encoder_encoder(self._regions(bottleneck))
        input_slice = self.memory.input_slices[self.memory_key]
        output_slice = self.memory.output_slices[self.memory_key]
        for _ in range(self.steps):
            readout = self.memory.voltage[..., output_slice].reshape(encoded.shape[0], -1)
            new_input = self.encoder_encoder_mlp(torch.cat([encoded, readout], dim=1))
            stimulus = torch.zeros_like(self.memory.voltage)
            stimulus[..., input_slice] = new_input.reshape(stimulus[..., input_slice].shape).to(stimulus)
            if self.grid_size is None:
                self.memory.step(stimulus * self.input_gain, dt=self.dt)
            else:
                # ALPACA accepts one state vector at a time; keep regions independent.
                memory = self.memory
                voltages, spikes = memory.voltage, memory.spikes
                states, outputs = [], []
                for region in range(len(voltages)):
                    memory.voltage, memory.spikes = voltages[region], spikes[region]
                    memory.step(stimulus[region] * self.input_gain, dt=self.dt)
                    states.append((memory.voltage, memory.spikes))
                    outputs.append(memory.y)
                memory.voltage = torch.stack([v for v, _ in states])
                memory.spikes = torch.stack([s for _, s in states])
                memory.y = {key: torch.stack([y[key] for y in outputs]) for key in memory.shape}
        return self.read(bottleneck)

    def read(self, bottleneck):
        output_slice = self.memory.output_slices[self.memory_key]
        readout = self.memory.voltage[..., output_slice]
        if self.grid_size is not None:
            readout = readout.reshape(*self.grid_size, -1)
        readout = readout.unsqueeze(0)
        return bottleneck + self.memory_decoder((bottleneck, readout))

class MemoryAdapter(nn.Module):
    """Store spatial observation state using ALPACA's neuron step."""

    width = 24
    steps = 4
    dt = 0.005

    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            AppendMemory(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.fusion[0].projection.weight.requires_grad_(False)
        self.drive = nn.Sequential(nn.Conv2d(130, self.width, 1), nn.GroupNorm(4, self.width))
        self.recurrent = nn.Conv2d(self.width, self.width, 3, padding=1, bias=False)
        self.readout = nn.Conv2d(2 * self.width, 128, 1, bias=False)
        nn.init.normal_(self.recurrent.weight, std=0.01)
        nn.init.zeros_(self.readout.weight)

        self.cells = Memory(high=1., low=-1., dims=1, size=1, device="cpu", interval=1, shape={}, name="spatial_grid")
        self.cells.input_mask.fill_(True)
        self.cells.requires_grad_(False)
        self.cells.threshold = 0.75
        self.cells.tau = -self.dt / math.log(0.65)
        self.state_shape = None

    def reset_memory(self):
        self.state_shape = None
        self.cells.voltage = self.cells.voltage.new_zeros(1)
        self.cells.spikes = self.cells.spikes.new_zeros(1)

    def detach_memory(self):
        self.cells.voltage = self.cells.voltage.detach()
        self.cells.spikes = self.cells.spikes.detach()

    def write(self, bottleneck, observed_fire_maps):
        """Write a usable observation, using the last active and burned maps."""
        if (observed_fire_maps is None or observed_fire_maps.ndim != 4
                or observed_fire_maps.shape[:2] != (bottleneck.shape[0], 2)):
            raise ValueError("observed_fire_maps must have shape [batch, 2, height, width]")
        maps = torch.nan_to_num(observed_fire_maps.to(bottleneck), nan=0)
        maps = F.adaptive_max_pool2d(maps, bottleneck.shape[-2:])
        drive = self.drive(torch.cat((bottleneck, maps), dim=1)) * 1.5
        if self.state_shape is None:
            self.state_shape = drive.shape
            self.cells.voltage = drive.new_zeros((1, drive.numel()))
            self.cells.spikes = torch.zeros_like(self.cells.voltage)
        elif self.state_shape != drive.shape:
            raise ValueError("Reset spatial memory before writing observations on a different grid")
        for _ in range(self.steps):
            spikes = self.cells.spikes.reshape(self.state_shape)
            current = drive + self.recurrent(spikes)
            self.cells.step(current.flatten()[None], dt=self.dt)

    def read(self, bottleneck):
        """Read the held spatial state without advancing the ALPACA neurons."""
        if self.state_shape is None:
            state = bottleneck.new_zeros((bottleneck.shape[0], 2 * self.width, *bottleneck.shape[-2:]))
        else:
            if self.state_shape[0] != bottleneck.shape[0]:
                raise ValueError("Reset spatial memory before changing the batch")
            state = torch.cat((self.cells.voltage.reshape(self.state_shape),
                               self.cells.spikes.reshape(self.state_shape)), dim=1)
        location = self.readout(state) + self.fusion[0].projection.bias[None, :, None, None]
        if location.shape[-2:] != bottleneck.shape[-2:]:
            location = F.interpolate(location, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        return bottleneck + self.fusion[1:](torch.cat((bottleneck, location), dim=1))

    def forward(self, bottleneck, observed_fire_maps):
        self.write(bottleneck, observed_fire_maps)
        return self.read(bottleneck)

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
