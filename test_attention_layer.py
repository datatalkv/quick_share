import math

import einops
import torch
from torch import nn, einsum, Tensor


class ContextAttention(nn.Module):
    r"""Attention mechanism inspired by `Hierarchical Attention Networks for
    Document Classification
    <https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf>`_
    """

    def __init__(self, input_dim: int, dropout: float, sum_along_seq: bool = False):
        super(ContextAttention, self).__init__()

        self.inp_proj = nn.Linear(input_dim, input_dim)
        self.context = nn.Linear(input_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sum_along_seq = sum_along_seq

    def forward(self, X: Tensor) -> Tensor:
        scores = torch.tanh_(self.inp_proj(X))
        attn_weights = self.context(scores).softmax(dim=1)
        self.attn_weights = attn_weights.squeeze(2)
        attn_weights = self.dropout(attn_weights)
        output = (attn_weights * X).sum(1) if self.sum_along_seq else (attn_weights * X)
        return output


class QueryKeySelfAttention(nn.Module):
    r"""Attention mechanism inspired by the well known multi-head attention. Here,
    rather than learning a value projection matrix that will be multiplied by
    the attention weights, we multiply such weights directly by the input
    tensor.

    The rationale behind this implementation comes, among other
    considerations, from the fact that Transformer based models tend to
    heavily overfit tabular. Therefore, by reducing the number of trainable
    parameters and multiply directly by the incoming tensor we help
    mitigating such overfitting
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float,
        use_bias: bool,
        n_heads: int,
    ):
        super(QueryKeySelfAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"

        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.qk_proj = nn.Linear(input_dim, input_dim * 2, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: seq length
        # l: target sequence length. Here l = s
        # m: used to refer indistinctively to s or l
        # h: number of attention heads,
        # d: head_dim
        q, k = self.qk_proj(X).chunk(2, dim=-1)
        q, k, x_rearr = map(
            lambda t: einops.rearrange(t, "b m (h d) -> b h m d", h=self.n_heads),
            (q, k, X),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, x_rearr)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)
        return output


if __name__ == "__main__":
    n_samp, n_seq, n_dim = 31, 21, 20
    x = torch.randn(n_samp, n_seq, n_dim)

    m_ca1 = ContextAttention(n_dim, dropout = 0.5, sum_along_seq=True)
    y_ca1 = m_ca1(x)
    print(f"ca1: {x.shape} --> {y_ca1.shape}")
    m_ca2 = ContextAttention(n_dim, dropout = 0.5, sum_along_seq=False)
    y_ca2 = m_ca2(x)
    print(f"ca2: {x.shape} --> {y_ca2.shape}")

    m_sa1 = QueryKeySelfAttention(n_dim, dropout=0.5, use_bias=False, n_heads=2)
    y_sa1 = m_sa1(x)
    print(f"sa1: {x.shape} --> {y_sa1.shape}")
    m_sa2 = QueryKeySelfAttention(n_dim, dropout=0.5, use_bias=False, n_heads=5)
    y_sa2 = m_sa2(x)
    print(f"sa2: {x.shape} --> {y_sa2.shape}")

    y_ca_sa = m_ca1(m_sa2(x))
    print(f"sa * ca: {x.shape} --> {y_ca_sa.shape}")
