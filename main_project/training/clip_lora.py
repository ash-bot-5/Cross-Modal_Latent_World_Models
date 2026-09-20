"""
LoRA building blocks for fine-tuning open_clip's ViT-L/14 image encoder.

Adapted from ~/CLIP-LoRA (github reference repo for few-shot CLIP classification),
reusing only the "split the fused nn.MultiheadAttention.in_proj_weight into per-
projection Linears, then LoRA-wrap them" mechanism. Everything else in that repo
(training loop, classification loss, optimizer/scheduler, eval harness) is specific
to few-shot image classification and is not reused here — see the module docstrings
below for the specific deviations.

Only the image tower (`clip_model.visual.transformer.resblocks`) is ever touched.
The text tower (`clip_model.transformer`) is left structurally untouched and frozen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps one existing nn.Linear with a frozen base + a trainable low-rank delta.

    Deliberately a pure functional forward (base + scaling * dropout(x) @ A^T @ B^T)
    with no in-place merge/unmerge of the base weight — unlike CLIP-LoRA's LinearLoRA,
    which mutates the "frozen" weight tensor in place around each forward call. That
    merge/unmerge trick is unsafe under autograd and especially risky combined with
    nn.DataParallel's module replication, so it is not ported here.
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(r)  # CLIP-LoRA's convention (not the more common alpha/r)

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B=0 at init -> LoRA delta is exactly 0, model starts identical to base CLIP

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base + delta


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for open_clip's ResidualAttentionBlock.attn.

    open_clip's ResidualAttentionBlock.attention() calls:
        self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
    where self.attn is a plain nn.MultiheadAttention(width, heads, batch_first=True)
    with a FUSED in_proj_weight/in_proj_bias (shape (3*embed_dim, embed_dim)) and a
    separate out_proj submodule — there are no standalone q_proj/k_proj/v_proj
    submodules to target directly. This class splits the fused in_proj weight/bias
    into three real nn.Linear layers (copying the pretrained weights exactly), wraps
    all four projections (q, k, v, out) with LoRALinear, and reimplements the
    forward pass via F.scaled_dot_product_attention (whose default 1/sqrt(head_dim)
    scale reproduces nn.MultiheadAttention's standard math exactly).

    Only need_weights=False is supported, matching the only way open_clip's vision
    tower ever calls attention.
    """

    def __init__(self, existing_mha: nn.MultiheadAttention, r: int, alpha: int, dropout: float):
        super().__init__()
        assert existing_mha.batch_first, "LoRAMultiheadAttention requires batch_first=True"
        assert existing_mha._qkv_same_embed_dim, "LoRAMultiheadAttention requires fused in_proj (self-attention)"

        embed_dim = existing_mha.embed_dim
        self.embed_dim = embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim = existing_mha.head_dim
        assert self.head_dim * self.num_heads == embed_dim

        in_w = existing_mha.in_proj_weight.data
        in_b = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

        def _split_linear(w_slice, b_slice):
            lin = nn.Linear(embed_dim, embed_dim, bias=b_slice is not None,
                             device=w_slice.device, dtype=w_slice.dtype)
            lin.weight.data.copy_(w_slice)
            if b_slice is not None:
                lin.bias.data.copy_(b_slice)
            return lin

        q_lin = _split_linear(in_w[:embed_dim], in_b[:embed_dim] if in_b is not None else None)
        k_lin = _split_linear(in_w[embed_dim:2 * embed_dim], in_b[embed_dim:2 * embed_dim] if in_b is not None else None)
        v_lin = _split_linear(in_w[2 * embed_dim:], in_b[2 * embed_dim:] if in_b is not None else None)

        self.q_proj = LoRALinear(q_lin, r, alpha, dropout)
        self.k_proj = LoRALinear(k_lin, r, alpha, dropout)
        self.v_proj = LoRALinear(v_lin, r, alpha, dropout)
        self.out_proj = LoRALinear(existing_mha.out_proj, r, alpha, dropout)

    def forward(self, query, key, value, need_weights: bool = False, attn_mask=None):
        assert not need_weights, "LoRAMultiheadAttention only supports need_weights=False"
        B, L, E = query.shape
        S = key.shape[1]

        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, L, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        attn_out = self.out_proj(attn_out)
        return attn_out, None


def apply_lora_to_visual_encoder(
    clip_model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.2,
) -> list[LoRAMultiheadAttention]:
    """Freezes the ENTIRE clip_model (both towers), then replaces `.attn` in every
    block of clip_model.visual.transformer.resblocks with a LoRAMultiheadAttention
    (q, k, v, out all LoRA-adapted). No positional subsetting — every block is
    touched unconditionally, matching the "all-layer coverage" requirement and
    sidestepping CLIP-LoRA's INDEX_POSITIONS_VISION table (which is keyed to
    OpenAI-CLIP backbone-name strings like 'ViT-L/14' incompatible with open_clip's
    'ViT-L-14' naming, and is unnecessary here anyway).

    clip_model.transformer (the TEXT tower) is left structurally untouched and
    frozen — this scopes fine-tuning to the image encoder only.

    Mutates clip_model in place. Does not call .train()/.eval() — mode management
    is the caller's responsibility. Returns the list of injected modules (for
    introspection/logging only).
    """
    for p in clip_model.parameters():
        p.requires_grad_(False)

    injected = []
    for i, block in enumerate(clip_model.visual.transformer.resblocks):
        assert isinstance(block.attn, nn.MultiheadAttention), f"block {i}.attn already replaced?"
        new_attn = LoRAMultiheadAttention(block.attn, r=r, alpha=alpha, dropout=dropout)
        block.attn = new_attn
        injected.append(new_attn)

    for block in clip_model.transformer.resblocks:  # text tower — sanity check only, untouched
        assert isinstance(block.attn, nn.MultiheadAttention)

    return injected


def lora_state_dict(clip_model: nn.Module) -> dict[str, torch.Tensor]:
    """Flat {dotted_param_name: cpu_tensor} for exactly the trainable LoRA A/B
    parameters in the image encoder. Excludes all frozen base weights (both the
    LoRALinear-wrapped ones and everything else in clip_model). Self-describing
    names (e.g. 'visual.transformer.resblocks.13.attn.q_proj.lora_A') — no external
    positional-index metadata needed, unlike CLIP-LoRA's layer_i-indexed scheme."""
    return {
        name: param.detach().cpu().clone()
        for name, param in clip_model.named_parameters()
        if param.requires_grad and "lora_" in name
    }


def load_lora_state_dict(clip_model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """In-place load onto a clip_model that already had apply_lora_to_visual_encoder
    called on it with matching r/alpha/dropout."""
    owned = dict(clip_model.named_parameters())
    missing = [k for k in state_dict if k not in owned]
    if missing:
        raise KeyError(f"LoRA state dict keys not found in model: {missing}")
    with torch.no_grad():
        for name, tensor in state_dict.items():
            owned[name].copy_(tensor.to(owned[name].device))
