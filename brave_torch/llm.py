import torch
from einops import rearrange
from torch import einsum, nn
from local_attention import LocalAttention
from zeta.nn import FeedForward
from brave_torch.main import BraveMultiModalFusion

# helpers


def exists(val):
    return val is not None


# normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    """RotaryEmbedding

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device):
        seq = torch.arange(
            max_seq_len, device=device, dtype=self.inv_freq.dtype
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    """Apply rotary positional embedding

    Args:
        pos (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# Transformer


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        window_size: int = 512,
        causal: bool = True,
        look_backward: int = 1,
        look_forward: int = 0,
        dropout: float = 0.1,
        shared_qk: bool = True,
        exact_window_size: bool = False,
        heads: int = None,
        dim_head: int = None,
        ff_mult=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                LocalAttention(
                    dim=dim,
                    window_size=window_size,
                    causal=causal,
                    look_backward=look_backward,
                    look_forward=look_forward,
                    dropout=dropout,
                    shared_qk=shared_qk,
                ),
            )

            self.ffn_layers.append(
                FeedForward(
                    dim=dim,
                    dim_out=dim,
                    mult=ff_mult,
                    dropout=dropout,
                ),
            )

    def forward(self, x):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = ffn(x) + x
            x = attn(x, x, x) + x
            x = ffn(x) + x
        return x


# classes


class LLM(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        num_of_vits: int = 4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = TransformerBlock(
            dim, depth, heads, dim_head, ff_mult
        )

        self.to_logits = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, num_tokens)
        )

        # BraveMultiModalFusion
        self.fuse = BraveMultiModalFusion(
            dim=dim,
            mult=4,
            image_size=image_size,
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            num_of_vits=num_of_vits,
        )

    def forward(self, x, img):
        x = self.emb(x)

        # BraveMultiModalFusion
        x = self.fuse(x, img)
        x = self.transformer(x)
        return self.to_logits(x)
