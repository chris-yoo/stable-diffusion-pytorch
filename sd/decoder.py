import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


# data 1 a1 a2 a3 a4 a5
# 이렇게 있을때 각 데이터별 feature 들의 평균과 분산으로 정규화를 하는 것은 layer norm
# Batch norm 은 배치의 데이터에 데이터의 각 feature 별로 정규화
# group norm은 layer norm 과 비슷하지만 모든 피처에 대해서 정규화를 하는게 아니라 그룹으로 쪼갠다
# 이게 효과가 있는 이유는 우리가 하고자 하는 연산은 conv 이다. 따라서 멀리 떨어져 있는 픽셀끼리는 연관이 없다 따라서 groupnorm 이 효과적이다.


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(
            32, channels
        )  # 입력채널을 32개의 그룹으로 나누겠다는 의미이다.
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):
        residue = x

        x = self.groupnorm(x)

        n, c, h, w = x.shape
        # (Batch_Size, Features, Height, Width) ->  x:(Batch_Size, Features, Height* Width)

        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) ->  x:(Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # x:(Batch_Size, Height * Width, Features) ->  x:(Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # x:(Batch_Size, Height * Width, Features) ->  x:(Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )  # filter의 개수만 바꿔줌

    def forward(self, x: torch.Tensor):
        # x:(Batch_Size, In_Channels, Height, Width)
        # VAE_ResidualBlock(128, 128) -> 차원이 같기 때문에 그냥 더해주면 됨
        # VAE_ResidualBlock(128, 256)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/8, Height/8) -> (Batch_Size, 512, Height/8, Height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/8, Height/8) -> (Batch_Size, 512, Height/4, Height/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/4, Height/4) -> (Batch_Size, 512, Height/2, Height/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (Batch_Size, 512, Height/2, Height/2) -> (Batch_Size, 256, Height, Height)
            nn.Upsample(scale_factor=2),  # 근처의 데이터를 복제함
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),  # 128개의 feature 를 32개로 나눈다.
            nn.SiLU(),
            # (Batch_Size, 128, Height, Height) -> (Batch_Size, 3, Height, Height)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # x->z:(Batch_Size, 4, Height/8, Height/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # z->x:(Batch_Size, 3, Height, Height)
        return x
