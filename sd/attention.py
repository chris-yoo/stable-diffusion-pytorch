import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(
            d_embed, 3 * d_embed, bias=in_proj_bias
        )  # 원래 Q,K,V 하나씩 각 임베딩 벡터로 프로젝션하여 생성해야 하지만 출력차원을 3배로 하고 프로젝션하면 더 효율적으로 할 수 있다.
        self.out_proj = nn.Linear(
            d_embed, d_embed, bias=out_proj_bias
        )  # 여기서 concatenate 한것을 선형변환을 한번 더 해준다 여기서 W0 는 learnable 하다.
        self.n_heads = n_heads
        self.d_head = (
            d_embed // n_heads
        )  # 위의 self.out_proj 의 부분의 in_channel 과 out_channel 이 같은 이유는 이런식으로 이미 각 헤드별 차원을 맞춰줬기 때문이다.

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x:(Batch_Size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, Dim*3) -> 3 tensor of shape (Batch_Size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)  # 마지막 차원을 기준으로 나눈다.

        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, H, Dim/H) -> (Batch_Size, H,Seq_len, Dim/H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (Batch_Size, H,Seq_len, Seq_len)
        # 행에 쿼리에 대한 정보가 있음
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper trianbles (above the principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(
                mask, -torch.inf
            )  # 마스킹 위치에 -∞ 를 채워둠 softmax 시 해당 부분이 exp(-inf) -> 0 이 되므로 확률이 사라짐

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_len, Seq_len) @ (Batch_Size, H, Seq_len, Dim/H) -> (Batch_Size, H, Seq_len, Dim/H)
        output = weight @ v

        # (Batch_Size, H, Seq_len, Dim/H) -> (Batch_Size, Seq_len, H, Dim/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output
