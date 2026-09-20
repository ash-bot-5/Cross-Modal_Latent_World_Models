"""
Rebalances the 2-GPU split for scripts that fully fine-tune both the CLIP image AND text
towers together (dynamics_model_clip_full_finetune.py,
dynamics_model_clip_full_finetune_combined.py,
dynamics_model_clip_full_finetune_combined_horizon16.py).

Problem: those scripts wrap only the image tower + MLP in nn.DataParallel(device_ids=[0,1])
(an even 50/50 image-batch split), then call encode_text() on the full statement table
separately afterward, always on device_ids[0], sequentially. GPU 1 sits idle for the entire
text phase. Measured (real CLIP ViT-L/14, micro_batch_size=256): 3.84s/step, only a 1.29x
speedup over a true single-GPU baseline (4.94s), because roughly half of every step runs
single-GPU with the other card idle.

Fix: give device_ids[1] a SMALLER image shard plus the full text tower, sized (via
calibrate_split, a one-time startup timing probe) so both GPUs finish at about the same
time each step -- instead of an even image split that leaves one GPU idle during text
encoding. Built entirely on torch.nn.parallel's own differentiable
scatter/replicate/parallel_apply/gather primitives (the same ones nn.DataParallel uses
internally), just with an uneven chunk_sizes split and an extra per-replica text-encode
branch (only one replica gets text_tokens != None) -- the same technique the long-standing
community "BalancedDataParallel" pattern uses for uneven-batch DataParallel, applied here to
a two-tower (image + text) step instead of a single tower.

Scope: intended for the TRAIN LOOP ONLY in each calling script. Validation, smoke_test, and
diagnostics keep using the existing, unmodified plain-DataParallel + sequential-encode_text
path in each script.

Precision note: torch.nn.parallel.parallel_apply only propagates a bool
(torch.is_autocast_enabled()) into its per-device worker threads, not the actual autocast
dtype -- each worker re-enters autocast with no explicit dtype, which defaults to float16
on CUDA. Confirmed empirically that this makes the existing scripts' DataParallel-wrapped
forward silently run in fp16 rather than the bf16 they intend. ImageTextJointStep.forward()
below works around this by re-entering torch.autocast(dtype=torch.bfloat16) explicitly
inside the module itself, so it doesn't depend on the calling thread's ambient autocast
state. This fix is deliberately scoped to this new module only -- the existing JointModel
classes in the calling scripts (used for validation/smoke_test/diagnostics/
--legacy_even_split) are left untouched, so --legacy_even_split remains a faithful
reproduction of each script's actual current (fp16-inside-DataParallel) behavior.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel._functions import Scatter, Gather


class ImageTextJointStep(nn.Module):
    """Wraps the SAME clip_model + mlp instances the calling script already constructed
    (not a copy) so replicate() can deep-copy them per-device fresh each step, exactly as
    nn.DataParallel does internally -- gradients accumulate back into the original
    parameters the same way DataParallel's replicate/backward already does today.
    text_tokens=None skips the text tower entirely for that replica; this is what lets one
    replica do image-only work while the other does image+text, via parallel_apply's
    per-replica kwargs_tup."""

    def __init__(self, clip_model: nn.Module, mlp: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = mlp

    def forward(
        self,
        pixel_values: torch.Tensor,
        actions: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Explicit dtype (not just "enabled") -- see module docstring precision note.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_t = self.clip_model.encode_image(pixel_values, normalize=True)
            z_pred = self.mlp(z_t, actions)
            unique_z = None
            if text_tokens is not None:
                unique_z = F.normalize(self.clip_model.encode_text(text_tokens), dim=-1)
        return z_pred, unique_z


def _median_forward_backward_time(fn, optimizer: torch.optim.Optimizer, n_warmup: int, n_probe_iters: int) -> float:
    """Median wall-clock time of n_probe_iters calls to fn() (each doing its own
    forward+backward), after n_warmup discarded warmup calls. Zeros grad after every call
    (including warmup) and the caller must never call optimizer.step() on this optimizer --
    calibration must not perturb the model's actual weights, only accumulate-then-discard
    gradients."""
    times = []
    for i in range(n_warmup + n_probe_iters):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.zero_grad()
        if i >= n_warmup:
            times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def calibrate_split(
    clip_model: nn.Module,
    mlp: nn.Module,
    device_ids: list[int],
    micro_batch_size: int,
    action_horizon: int,
    global_tokens: torch.Tensor,
    n_warmup: int = 2,
    n_probe_iters: int = 3,
) -> dict | None:
    """One-time startup probe (real clip_model/mlp, real micro_batch_size/action_horizon/
    statement table, dummy random pixel/action data) that measures image-only cost on
    device_ids[0] and text-only cost on device_ids[1] in isolation, then solves for an
    integer image-batch split (n0, n1) so both GPUs finish at about the same wall-clock
    time once device_ids[1] also carries the text tower.

    Returns a dict {"n0", "n1", "t_img_full", "t_text", "device_ids"} on success, or None
    (caller should fall back to the plain even-split DataParallel path for the whole run)
    if anything raises -- e.g. an OOM during the probe itself.

    IMPORTANT: never calls optimizer.step() -- only forward+backward+zero_grad, so the
    model's actual weights (random init or resumed) are untouched by calibration.
    """
    try:
        dev0, dev1 = device_ids[0], device_ids[1]
        params = list(clip_model.parameters()) + list(mlp.parameters())
        probe_optimizer = torch.optim.SGD(params, lr=0.0)

        pixel_values = torch.randn(micro_batch_size, 3, 224, 224, device=dev0)
        actions = torch.randn(micro_batch_size, action_horizon, 2, device=dev0)

        def image_only():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_t = clip_model.encode_image(pixel_values, normalize=True)
                z_pred = mlp(z_t, actions)
            z_pred.float().sum().backward()

        t_img_full = _median_forward_backward_time(image_only, probe_optimizer, n_warmup, n_probe_iters)

        # clip_model lives on dev0 -- encode_text needs a dev1-resident copy to run there at
        # all (matches how run_split_step's degenerate branch does this too: replicate()
        # broadcasts FROM devices[0], so dev0 must come first even though only the dev1
        # replica is used). Re-replicated fresh each timed call, same as the real per-step
        # cost run_split_step pays.
        tokens_dev1 = global_tokens.to(dev1)

        def text_only():
            text_replica = replicate(clip_model, [dev0, dev1])[1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unique_z = F.normalize(text_replica.encode_text(tokens_dev1), dim=-1)
            unique_z.float().sum().backward()

        t_text = _median_forward_backward_time(text_only, probe_optimizer, n_warmup, n_probe_iters)

        c = t_img_full / micro_batch_size  # measured per-image cost
        n0 = round((micro_batch_size + t_text / c) / 2)
        n0 = max(1, min(micro_batch_size, n0))
        n1 = micro_batch_size - n0

        result = {
            "n0": n0, "n1": n1, "t_img_full": t_img_full, "t_text": t_text,
            "device_ids": [dev0, dev1],
        }
        print(f"[gpu_split_joint_step] calibration: T_img_full={t_img_full:.3f}s "
              f"T_text={t_text:.3f}s -> split n0={n0} (GPU {dev0}, image-only) / "
              f"n1={n1} (GPU {dev1}, image+text)")
        return result
    except Exception as e:
        print(f"[gpu_split_joint_step] WARNING: calibration failed ({type(e).__name__}: {e}), "
              f"falling back to even-split DataParallel for this run.")
        return None


def run_split_step(
    step_module: ImageTextJointStep,
    pixel_values: torch.Tensor,
    actions: torch.Tensor,
    global_tokens_gpu1: torch.Tensor,
    device_ids: list[int],
    n0: int,
    n1: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One training step's forward pass: pixel_values/actions are scattered unevenly
    (n0 on device_ids[0], n1 on device_ids[1]) so device_ids[1] gets a SMALLER image shard
    and ALSO runs the full text tower. parallel_apply launches one thread per device and
    doesn't synchronize between them until both finish, so this is genuine hardware
    overlap (device_ids[1]'s smaller image shard + text running concurrently with
    device_ids[0]'s larger image-only shard), not just sequential bookkeeping.

    Returns (z_pred, unique_z), both on device_ids[0], ready to feed into the existing
    infonce_logits/loss code unchanged. No outer `torch.autocast(...)` block is needed at
    the call site -- ImageTextJointStep.forward() (and this function's own degenerate
    branch) already re-enter bf16 autocast explicitly themselves (see module docstring's
    precision note), so this function is self-contained on precision regardless of the
    caller's ambient autocast state.
    """
    dev0, dev1 = device_ids[0], device_ids[1]

    if n1 == 0:
        # Degenerate case: all images on dev0 alone, text alone on dev1, no scatter/gather
        # machinery needed for the image side. Still runs concurrently -- these are two
        # independently-issued device streams. replicate() broadcasts FROM devices[0], so
        # it must be called with dev0 first (step_module's params live there) even though
        # only the dev1 replica is used here -- replicas[0] is simply unused.
        text_replica = replicate(step_module, device_ids)[1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unique_z_dev1 = F.normalize(text_replica.clip_model.encode_text(global_tokens_gpu1), dim=-1)
            z_t = step_module.clip_model.encode_image(pixel_values, normalize=True)
            z_pred = step_module.mlp(z_t, actions)
        unique_z = unique_z_dev1.to(dev0)
        return z_pred, unique_z

    replicas = replicate(step_module, device_ids)
    chunk_sizes = [n0, n1]
    pixel_chunks = Scatter.apply(device_ids, chunk_sizes, 0, pixel_values)
    action_chunks = Scatter.apply(device_ids, chunk_sizes, 0, actions)

    inputs = [(pixel_chunks[0], action_chunks[0]), (pixel_chunks[1], action_chunks[1])]
    kwargs_tup = [{"text_tokens": None}, {"text_tokens": global_tokens_gpu1}]
    outputs = parallel_apply(replicas, inputs, kwargs_tup, device_ids)

    z_pred = Gather.apply(dev0, 0, outputs[0][0], outputs[1][0])
    unique_z = outputs[1][1].to(dev0)
    return z_pred, unique_z
