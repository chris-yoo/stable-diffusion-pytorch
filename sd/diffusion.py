import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

# torch.nn.fucntional 에는 activation function, loss function, 등이 있다.
# torch.nn 모듈에서 제공하는 클래스(예: nn.ReLU(), nn.Softmax())는 객체를 생성해야 사용 가능
# F 모듈에서는 함수로 바로 사용할 수 있음

# relu = nn.ReLU()  # 객체 생성
# output = relu(x)  # 사용

# output = F.relu(x)  # 바로 사용 가능


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, n_embd * 4)
        self.linear_2 = nn.Linear(n_embd * 4, n_embd * 4)

    def forward(self, x):
        # x: (1, 320)
        # x: (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        # 비선형성 추가
        x = F.silu(x)
        # x: (1, 1280) -> (1, 1280)
        x = self.linear_2(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature, time):
        # feature: (batch, in_channels, height, width)
        # time: (1,1280)
        # time embedding 을 더해주는 방식으로 작동한다.

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(
            -1
        )  # 마지막차원을 두번 추가하면서 feature 와 차원이 같아진다.

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embc: int, d_context=768):
        super().__init__()
        channels = n_head + n_embc
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self-Attention with skip connection

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x

        # Normaliazation + Cross Attention with skip connection
        x = self.layernorm_2(x)

        # Cross Attention
        x = self.attention_2(x, context)

        x += residue_short

        # Normalization + FF with GeGLU and skip connection

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(
            2, dim=-1
        )  # 애초에 channel 차원을 8배로 늘리고 마지막 차원 즉 channel에 대해서 두개로 나누어 줘서 각각 x 와 gate 로 나누어 준다.

        x = x * F.gelu(gate)  # 각 요소별 곱이다. matrix multiplication 이 아니다.

        x = self.linear_geglu_2(x)

        x += residue_short

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch, features, height, width) -> (batch, features, height * 2, width * 2)
        x = F.interpolate(
            x, scale_factor=2, mode="nearesr"
        )  # nn.Upsample(scale_factor = 2) 랑 같은 연산이다.
        return self.conv(x)


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(
                layer, UNET_AttentionBlock
            ):  # context 정보와 x 를 cross attention 하기 위해서
                # isinstance(object, class)는 객체(object)가 특정 클래스(class)의 인스턴스인지 확인하는 함수입니다.

                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # layer 가 UNET_ResidualBlock 의 인스턴스이면 해당 time 정보와 같이 파라미터입력
                x = layer(x, time)
            else:
                x = layer(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 부분은 이미지의 크기는 줄이고 특징은 계속 늘린다.
        self.encoders = nn.ModuleList(
            [
                # (batch, 4, height/8, width/8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # (batch, 320, height/8, width/8) ->  (batch, 4, height/16, width/16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
                ),
                # (batch, 640, height/16, width/16) ->  (batch, 640, height/32, width/32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (batch, 1280, height/32, width/32) ->  (batch, 1280, height/64, width/64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                # (batch, 1280, height/64, width/64) -> (batch, 1280, height/64, width/64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        # 왜 bottleneck 의 출력값의 차원이 1280 인데 다음 디코더의 입력 차원이 두배가 된것인가? -> skip connection 으로 연결시 차원이 두배가 된다.
        self.decoders = nn.ModuleList(
            [
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    UpSample(1280),
                ),
                # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    UpSample(640),
                ),
                # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
            ]
        )


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(
            32, in_channels
        )  # in_channels 를 32개의 그룹으로 나눈다.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch, 320, height/8, width/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        # (batch, 320, height/8, width/8) -> (batch, 4 height/8, width/8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor):
        # latent z: (Batch_Size, 4, Height/8, Width/8)
        # context: (Batch_Size, 4, Seq_Len, Dim)
        # time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height /8, Width/8) -> (Batch, 320, Height /8, Width/8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height /8, Width/8) -> (Batch, 4, Height /8, Width/8) input 이랑 output Dimension 이 같아야 한다.
        output = self.final(output)
