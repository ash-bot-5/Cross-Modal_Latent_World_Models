"""
Numerical-equivalence check for gpu_split_joint_step.py's asymmetric GPU-split forward
pass, run against a real CLIP ViT-L/14 (both towers unfrozen, matching the full-finetune
scripts' configuration).

What this checks (and why it's NOT simply "old production path vs. new split path"):
CLIP ViT/LayerNorm has no batch-dependent statistics (no BatchNorm), so splitting a batch
across devices and gathering the results back should reproduce the SAME per-sample outputs
as a single, unsplit forward pass -- up to ordinary bf16 floating-point noise. That single-
GPU, single-forward-pass, explicit-bf16 computation is used here as the REFERENCE ("ground
truth"), and the new split-step path (both the normal interior-split branch and the
degenerate n1=0 branch) is checked against it, for both outputs and backward gradients.

The EXISTING production path (JointModel + nn.DataParallel + sequential encode_text) is
deliberately NOT used as the reference: per gpu_split_joint_step.py's docstring, that path
has a pre-existing bug where torch.nn.parallel.parallel_apply silently runs each replica's
autocast region in float16 rather than the intended bfloat16 (confirmed empirically during
this project's GPU-rebalancing work). This test still runs that legacy path and reports
(not asserts) its divergence from the bf16 reference, to keep that finding's magnitude
documented -- but it is expected to diverge more than the new path, by design, since the
new path's precision bug is intentionally fixed while the legacy path's is deliberately
left as-is (see gpu_split_joint_step.py's module docstring for why).

Run: python test_gpu_split_equivalence.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from dynamics_model import LatentDynamicsModel
from gpu_split_joint_step import ImageTextJointStep, run_split_step

MICRO_BATCH_SIZE = 32
ACTION_HORIZON = 8
N_TEXT_STATEMENTS = 64
ATOL = 2e-2  # bf16-appropriate tolerance for L2-normalized 768-dim vectors
RTOL = 2e-2


class LegacyJointModel(nn.Module):
    """Minimal reproduction of the existing production scripts' JointModel (image tower +
    MLP only; text encoded separately). Deliberately does NOT apply the explicit-bf16 fix
    -- this mirrors today's actual (buggy) behavior on purpose, so its divergence from the
    bf16 reference below can be measured and reported."""

    def __init__(self, clip_model: nn.Module, mlp: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = mlp

    def forward(self, pixel_values, actions):
        z_t = self.clip_model.encode_image(pixel_values, normalize=True)
        return self.mlp(z_t, actions)


def sample_named_grads(clip_model: nn.Module, mlp: nn.Module) -> dict[str, torch.Tensor]:
    """One representative gradient each from the image tower, text tower, and MLP."""
    image_name = next(n for n, p in clip_model.named_parameters() if n.startswith("visual.") and p.requires_grad)
    text_name = next(n for n, p in clip_model.named_parameters()
                      if not n.startswith("visual.") and n not in ("logit_scale", "logit_bias") and p.requires_grad)
    named = dict(clip_model.named_parameters())
    grads = {
        f"clip_model.{image_name}": named[image_name].grad,
        f"clip_model.{text_name}": named[text_name].grad,
        "mlp.fc1.weight": mlp.fc1.weight.grad,
    }
    return {k: (v.clone() if v is not None else None) for k, v in grads.items()}


def zero_all_grads(clip_model: nn.Module, mlp: nn.Module) -> None:
    for p in clip_model.parameters():
        p.grad = None
    for p in mlp.parameters():
        p.grad = None


def compare(name: str, a: torch.Tensor, b: torch.Tensor, assert_close: bool) -> float:
    a, b = a.float(), b.float()
    max_abs_diff = (a - b).abs().max().item()
    close = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    status = "OK" if close else ("DIVERGES (expected)" if not assert_close else "FAIL")
    print(f"  {name}: max_abs_diff={max_abs_diff:.4e} allclose={close} [{status}]")
    if assert_close:
        assert close, f"{name} diverged beyond tolerance: max_abs_diff={max_abs_diff:.4e}"
    return max_abs_diff


def main():
    if torch.cuda.device_count() < 2:
        print(f"SKIP: this test requires 2 CUDA devices, found {torch.cuda.device_count()}.")
        return

    device_ids = [0, 1]
    dev0 = device_ids[0]
    torch.manual_seed(0)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
    clip_model = clip_model.to(dev0).eval()  # eval(): removes any dropout as a confound, see module docstring
    for n, p in clip_model.named_parameters():
        p.requires_grad_(n not in ("logit_scale", "logit_bias"))

    mlp = LatentDynamicsModel(horizon=ACTION_HORIZON).to(dev0).eval()

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    strings = [f"The block is touching the other block. The peg is touching block {i % 8}."
               for i in range(N_TEXT_STATEMENTS)]
    global_tokens = tokenizer(strings).to(dev0)
    global_tokens_gpu1 = global_tokens.to(device_ids[1])

    torch.manual_seed(1)
    pixel_values = torch.randn(MICRO_BATCH_SIZE, 3, 224, 224, device=dev0)
    actions = torch.randn(MICRO_BATCH_SIZE, ACTION_HORIZON, 2, device=dev0)

    # ---- reference: single GPU, single forward pass, explicit bf16 ----
    zero_all_grads(clip_model, mlp)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_t_ref = clip_model.encode_image(pixel_values, normalize=True)
        z_pred_ref = mlp(z_t_ref, actions)
        unique_z_ref = F.normalize(clip_model.encode_text(global_tokens), dim=-1)
    (z_pred_ref.float().sum() + unique_z_ref.float().sum()).backward()
    grads_ref = sample_named_grads(clip_model, mlp)
    z_pred_ref, unique_z_ref = z_pred_ref.detach(), unique_z_ref.detach()

    # ---- new path: interior split (n0, n1 both > 0) ----
    step_module = ImageTextJointStep(clip_model, mlp)
    zero_all_grads(clip_model, mlp)
    n0, n1 = 24, MICRO_BATCH_SIZE - 24
    z_pred_interior, unique_z_interior = run_split_step(
        step_module, pixel_values, actions, global_tokens_gpu1, device_ids, n0, n1
    )
    (z_pred_interior.float().sum() + unique_z_interior.float().sum()).backward()
    grads_interior = sample_named_grads(clip_model, mlp)
    z_pred_interior, unique_z_interior = z_pred_interior.detach(), unique_z_interior.detach()

    # ---- new path: degenerate split (n1 == 0) ----
    zero_all_grads(clip_model, mlp)
    z_pred_degen, unique_z_degen = run_split_step(
        step_module, pixel_values, actions, global_tokens_gpu1, device_ids, MICRO_BATCH_SIZE, 0
    )
    (z_pred_degen.float().sum() + unique_z_degen.float().sum()).backward()
    grads_degen = sample_named_grads(clip_model, mlp)
    z_pred_degen, unique_z_degen = z_pred_degen.detach(), unique_z_degen.detach()

    # ---- legacy path: existing production pattern (DataParallel + sequential encode_text) ----
    legacy_model = LegacyJointModel(clip_model, mlp)
    ddp_legacy = nn.DataParallel(legacy_model, device_ids=device_ids)
    zero_all_grads(clip_model, mlp)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_pred_legacy = ddp_legacy(pixel_values, actions)
        unique_z_legacy = F.normalize(clip_model.encode_text(global_tokens), dim=-1)
    (z_pred_legacy.float().sum() + unique_z_legacy.float().sum()).backward()
    grads_legacy = sample_named_grads(clip_model, mlp)
    z_pred_legacy, unique_z_legacy = z_pred_legacy.detach(), unique_z_legacy.detach()

    print(f"\n=== Interior split (n0={n0}, n1={n1}) vs. bf16 reference [must match] ===")
    compare("z_pred", z_pred_interior, z_pred_ref, assert_close=True)
    compare("unique_z", unique_z_interior, unique_z_ref, assert_close=True)
    for k in grads_ref:
        compare(f"grad[{k}]", grads_interior[k], grads_ref[k], assert_close=True)

    print(f"\n=== Degenerate split (n0={MICRO_BATCH_SIZE}, n1=0) vs. bf16 reference [must match] ===")
    compare("z_pred", z_pred_degen, z_pred_ref, assert_close=True)
    compare("unique_z", unique_z_degen, unique_z_ref, assert_close=True)
    for k in grads_ref:
        compare(f"grad[{k}]", grads_degen[k], grads_ref[k], assert_close=True)

    print(f"\n=== Legacy production path (fp16-inside-DataParallel bug, NOT fixed) vs. bf16 reference "
          f"[informational only, larger divergence expected] ===")
    compare("z_pred", z_pred_legacy, z_pred_ref, assert_close=False)
    compare("unique_z", unique_z_legacy, unique_z_ref, assert_close=False)
    for k in grads_ref:
        compare(f"grad[{k}]", grads_legacy[k], grads_ref[k], assert_close=False)

    print("\nPASS: new split-step path (interior and degenerate) matches the bf16 reference "
          "within tolerance for both outputs and gradients.")


if __name__ == "__main__":
    main()
