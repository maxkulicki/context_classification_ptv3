"""
CLS-attention multi-source context encoder for Phase 2.

Architecture:
    Each of the 4 sources passes through its own encoder (shared with Phase 1)
    producing a 256-dim source token. A learned [CLS] token aggregates across
    all available source tokens via 1-layer multi-head self-attention.

    Source dropout is applied per-sample during training: dropped sources are
    excluded from the CLS attention via key_padding_mask (their encoder output
    is still computed for the auxiliary classification heads in the parent model).

    Missing/dropped sources never need zero-filling — the attention mask handles
    variable availability cleanly.

Source dropout curriculum (Phase 2):
    p_all ramps linearly from 0 at epoch 0 to p_all_end at warmup_epochs, then holds.
    The k ∈ {1,2,3} branch is always uniform over k.

    epoch 0:   k=1 33%, k=2 33%, k=3 33%, k=4  0%
    epoch 20:  k=1 28%, k=2 28%, k=3 28%, k=4 18%
    epoch 40+: k=1 22%, k=2 22%, k=3 22%, k=4 35%  ← target
"""

import random

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model


def _sample_source_mask(
    B: int,
    device: torch.device,
    epoch,
    p_all_end: float = 0.35,
    warmup_epochs: int = 40,
) -> torch.BoolTensor:
    """Sample a per-sample source activation mask.

    Parameters
    ----------
    B : int
        Batch size.
    device : torch.device
        Target device for the returned tensor.
    epoch : int or None
        Current training epoch (0-indexed). None → use full target distribution
        (p_all = p_all_end), which is the correct behaviour at inference time.
    p_all_end : float
        Probability of activating all 4 sources at the end of the curriculum.
    warmup_epochs : int
        Number of epochs over which p_all ramps from 0 to p_all_end.

    Returns
    -------
    torch.BoolTensor of shape (B, 4), True = source is active.
    Source order: ['ae', 'topo', 'sinr', 'gpn'].
    """
    if epoch is None:
        p_all = p_all_end
    else:
        t = min(epoch / warmup_epochs, 1.0)
        p_all = t * p_all_end

    n = 4
    rows = []
    for _ in range(B):
        if random.random() < p_all:
            rows.append([True, True, True, True])
        else:
            k = random.randint(1, n - 1)  # uniform over {1, 2, 3}
            idx = random.sample(range(n), k)
            row = [False] * n
            for i in idx:
                row[i] = True
            rows.append(row)

    return torch.tensor(rows, dtype=torch.bool, device=device)


@MODELS.register_module("CLSCtxEncoder-v1m1")
class CLSContextEncoder(nn.Module):
    """CLS-attention aggregator for all 4 context sources.

    Each source is encoded by its own Phase-1 encoder into a 256-dim token.
    A learned [CLS] token aggregates across active source tokens via 1-layer MHSA.
    Dropped sources are masked out of the attention; their tokens are still returned
    for auxiliary classification heads in the parent model.

    Parameters
    ----------
    ae_cfg, topo_cfg, sinr_cfg, gpn_cfg : dict
        Configs for per-source encoders (built via MODELS registry).
    embed_dim : int
        Attention and output dimensionality (256).
    num_heads : int
        Number of attention heads (4).
    """

    # Fixed ordering — must match the 4-bit active_mask columns
    SOURCE_ORDER = ["ae", "topo", "sinr", "gpn"]

    def __init__(
        self,
        ae_cfg: dict,
        topo_cfg: dict,
        sinr_cfg: dict,
        gpn_cfg: dict,
        embed_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoders = nn.ModuleDict(
            {
                "ae": build_model(ae_cfg),
                "topo": build_model(topo_cfg),
                "sinr": build_model(sinr_cfg),
                "gpn": build_model(gpn_cfg),
            }
        )

        # Learned [CLS] token, shape (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        source_dict: dict,
        active_mask=None,
    ):
        """
        Parameters
        ----------
        source_dict : dict[str, Tensor]
            {'ae': (B, 64), 'topo': (B, 6), 'sinr': (B, 256), 'gpn': (B, 18)}
        active_mask : BoolTensor of shape (B, 4) or None
            True = source is active (included in CLS attention).
            None = all sources active.

        Returns
        -------
        ctx : Tensor (B, embed_dim)
            CLS token output — the aggregated context embedding.
        source_tokens : dict[str, Tensor (B, embed_dim)]
            Per-source embeddings (always computed, for auxiliary heads).
        """
        B = next(iter(source_dict.values())).shape[0]

        # 1. Encode all sources (always — aux heads need gradient regardless of masking)
        source_tokens = {
            name: self.encoders[name](source_dict[name])
            for name in self.SOURCE_ORDER
        }  # each (B, 256)

        # 2. Build token sequence: [CLS, ae, topo, sinr, gpn] → (B, 5, 256)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 256)
        src_seq = torch.stack(
            [source_tokens[n] for n in self.SOURCE_ORDER], dim=1
        )  # (B, 4, 256)
        tokens = torch.cat([cls, src_seq], dim=1)  # (B, 5, 256)

        # 3. Build key_padding_mask: (B, 5), True = position is IGNORED by attention
        if active_mask is not None:
            # CLS is always active (False = not masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            # Dropped sources → True (masked out of attention)
            src_mask = ~active_mask  # (B, 4)
            pad_mask = torch.cat([cls_mask, src_mask], dim=1)  # (B, 5)
        else:
            pad_mask = None

        # 4. Self-attention — CLS attends to all active source tokens
        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=pad_mask)

        # 5. CLS output → context embedding
        ctx = self.norm(attn_out[:, 0])  # (B, embed_dim)

        return ctx, source_tokens
