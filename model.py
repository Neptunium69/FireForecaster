import torch
import torch.nn as nn
from utae import UTAE
from alpaca.moc import Memory

class FireForecast(nn.Module):
    def __init__(self,input_channels) -> None:
        super().__init__()
        self.encoder=FireHistoryEncoder(input_channels=input_channels)


class FireHistoryEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.utae = UTAE(
            input_dim=input_channels,
            encoder_widths=[32, 32, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            n_head=8,
            d_model=128,
            d_k=4,
            agg_mode="att_group",
            encoder=True,
        )

    def forward(self, history, dates):
        features, multiscale_features = self.utae(
            history,
            batch_positions=dates,
        )
        return features

class AppendMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(512, 32)

    def forward(self, inputs):
        features, memory = inputs
        height, width = features.shape[-2:]

        memory = self.projection(memory)[:, :, None, None]
        memory = memory.expand(-1, -1, height, width)

        return torch.cat([features, memory], dim=1)

class MemoryAdapter(nn.Module):
    def __init__(self,high,low,dims,size,device,shape) -> None:
        self.memory=Memory(high,low,dims,size,device,interval=1,shape=shape,name="memory")
        self.encoder_encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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
        self.memory_decoder = nn.Sequential(
            AppendMemory(),  # ([B, 32, H, W], [B, 512]) -> [B, 64, H, W]

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )