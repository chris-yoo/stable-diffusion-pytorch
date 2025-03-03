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
        # nn.Linear 의 bias property 는 xwt+b 라는 선형변환에서 b의 항을 추가할 것인지 설정할 수 있는 것이다. 기본값은 True 이다.
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_pro_bias=True,
        out_proj_bias=True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_pro_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_pro_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_pro_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = (
            d_embed // n_heads
        )  # Dim 을 헤드수로 나누어 multihead attention 을 하기 위해서

    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # -1 을 넣는 이유는 해당 차원을 자동으로 계산하기 위해서이다. sequence_length 에 해당하는 부분이다.
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by wq
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        # view 라는 것은 기존 메모리의 순서를 유지하면서 변환하기 때문에 1번째 차원을 1번째 2번째 차원으로 쪼개는 것은 가능, but 쪼개서 0번째 2번째로 transpose 는 불가능 따라서 view 를 해주고 transpose를 한번 해줘야 한다.

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        # Attention Scores=Q×KT  k.transpose(-1, -2) 가 kt 의역할을 한다.
        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
