import torch
from torch import nn
from torch.nn import fucntional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # look up table 을 생성
        # (n_vocab, n_embd) 행렬을 생성
        # 1번 토큰은 1번토큰에 해당하는 vector를 반환한다.
        # nn.Linear 은 projection 하는 거고 이거는 값을 넣으면 사전처럼 바로 찾아주는거다

        self.position_embedding = nn.Parameter(
            torch.zeros(n_tokens, n_embd)
        )  # 토큰개수와 임베딩 차원, 0으로 초기화 하고 학습가능한 변수이다.

    def forward(self, tokens):
        # (Batch_Size, Seq_len) -> (Batch_Size, Seq_len, Dim
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor):
        # (Batch_Size, Seq_Len, Dim)
        residue = x

        ## SELF ATTENTION

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        ## FEEDFORWARD LAYER

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # Quick activation function
        x = self.linear_2(x)
        x += residue


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        # super 이 없어도 사실 nn 모듈을 사용할 수 있긴한다. 하지만 있는것이 더 권장된다.
        self.embedding = CLIPEmbedding(
            49408, 768, 77
        )  # 49408: 클립에 사용되는 어휘의 개수, 768 은 각 토큰이 임베딩 되는 차원, 77은 최대 토큰 개수(한번에 처리할 수 있는)

        self.layers = nn.ModuleList(
            [CLIPLayer(12, 768) for i in range(12)]
        )  # transformer 구조를 12번 통과한다.

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)  # 토큰을 768차원의 벡터로 변환

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
