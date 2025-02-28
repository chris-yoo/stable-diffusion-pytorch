import torch
from torch import nn
from torch.nn import funcional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        # nn.Sequentail 의 생성자 호출
        super().__init__(
            # 이미지의 크기는 줄이고 feature의 수는 늘린다. pixel 수를 줄이고 각 pixel 이 가지는 의미가 많아진다.(feature 수의 증가)
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # Attention Blocks
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            # Residual Block
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # Group Normalization
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.SiLU(),
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.SiLU(),
            # bottle neck
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    # tipe hint 가 포함되어있음
    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        for module in self:  # 레이어를 순차적으로 적용
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # 차원을 맞추기 위해서 패딩을 추가
            x = module(x)

        # (Batch_Size, 8, Height/8, Width/8) -> two tensor of shape (Batch_size,4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # 차원 1 을 기준으로 두개로 나눈다.

        # (Batch_size,4, Height/8, Width/8) -> (Batch_size,4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # 최대값과 최소값을 수치적으로 고정 -> hyper parameter?

        # (Batch_size,4, Height/8, Width/8) -> (Batch_size,4, Height/8, Width/8)
        variance = log_variance.exp()

        stdev = variance.sqrt()

        # z=N(0,1) -> x=N(mean, std)
        # x =  mean + std*z
        x = mean + stdev * noise
        x *= 0.18215

        return x
