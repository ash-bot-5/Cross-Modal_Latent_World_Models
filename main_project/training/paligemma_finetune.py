"""
Fine-tune PaliGemmaWM into a latent dynamics model for LangTable.

Motivation: we are replacing the FiLM-MLP dynamics model (dynamics_model.py,
LatentDynamicsModel: (z_t, actions) -> z_pred via a small conditioned MLP) with a
fine-tuned PaliGemma that predicts future semantic latents directly from pixels.
Input: current frame + action sequence. A single learnable `rand_n` token is
appended to the end of the image+action token sequence; we extract its final
hidden state as z_pred, project it to 768-dim, and train via InfoNCE (hard
negatives only, no in-batch negatives) to align with z_sem — CLIP ViT-L/14 Q&A
embeddings of the same (image, question) pairs used to train the FiLM model, so
the two dynamics models are directly comparable. LoRA on the Gemma attention
projections makes this tractable on 2x RTX 3090 Ti.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/training/paligemma_finetune.py
"""

import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from swm.paligemma_wm import PaliGemmaWMProcessor
from swm.paligemma_wm.configuration_paligemma_wm import PaliGemmaWMConfig
from swm.paligemma_wm.modeling_paligemma_wm import PaliGemmaWMModel

CHECKPOINT = "jacob3333/paligemma-3b-mix-224-wm"
ACTION_DIM = 2
Z_SEM_DIM = 768


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PaliGemmaLatentDynamics(PaliGemmaWMModel):
    """PaliGemmaWM base model + a learnable readout token for latent prediction.

    Forward pass builds [image tokens][action tokens][rand_n], runs it through
    the parent PaliGemmaWMModel, and reads out rand_n's final hidden state.
    """

    def __init__(self, config: PaliGemmaWMConfig):
        super().__init__(config)
        self.rand_n = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.projection_head = nn.Linear(config.hidden_size, Z_SEM_DIM, bias=False)

    def forward(self, pixel_values: torch.Tensor, action_values: torch.Tensor) -> torch.Tensor:
        device = pixel_values.device
        B = pixel_values.shape[0]
        num_img_tok = self.config.text_config.num_image_tokens
        num_act_tok = action_values.shape[1]

        input_ids = torch.cat(
            [
                torch.full((B, num_img_tok), self.config.image_token_id, dtype=torch.long, device=device),
                torch.full((B, num_act_tok), self.config.action_token_id, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        rand_n_tok = self.rand_n.to(inputs_embeds.dtype).expand(B, -1, -1)
        inputs_embeds = torch.cat([inputs_embeds, rand_n_tok], dim=1)

        # The whole sequence is a single prefix block (no suffix/label), so it
        # gets PaliGemma's bidirectional-prefix attention end to end; `rand_n`,
        # appended last, has unmasked access to every image/action position.
        token_type_ids = torch.zeros(B, num_img_tok + num_act_tok + 1, dtype=torch.long, device=device)
        attention_mask = torch.ones(B, num_img_tok + num_act_tok + 1, dtype=torch.long, device=device)

        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            action_values=action_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        z_pred = self.projection_head(outputs.last_hidden_state[:, -1, :])
        return F.normalize(z_pred, dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PaliGemmaLangTableDataset(Dataset):
    HORIZON = 8

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        self.frames: list = []
        self.actions: list[np.ndarray] = []
        self.z_sem: list[np.ndarray] = []
        self.unique_zsem: list[torch.Tensor] = []
        self.neg_indices: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []

        pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))

        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            zsem_path = pkl_path.replace(".pkl", "_zsem.npz")
            zhard_path = pkl_path.replace(".pkl", "_zhard.npz")

            if not os.path.exists(zsem_path):
                warnings.warn(f"Missing zsem file for {pkl_file}, skipping.")
                continue
            if not os.path.exists(zhard_path):
                warnings.warn(f"Missing zhard file for {pkl_file}, skipping episode.")
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)

            z_sem_data = np.load(zsem_path)["z_sem"]  # (T, 768)
            zhard = np.load(zhard_path)
            T = len(z_sem_data)

            ep_idx = len(self.actions)
            self.frames.append(ep["frames"])
            self.actions.append(ep["actions"])
            self.z_sem.append(z_sem_data)
            self.unique_zsem.append(torch.tensor(zhard["unique_zsem"], dtype=torch.float32))  # (N_ep, 768)
            self.neg_indices.append(zhard["neg_indices"])  # (T, 224) int16, episode-local

            for t in range(T - self.HORIZON):
                self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.indices[idx]

        frame = np.asarray(self.frames[ep_idx][t])  # (H, W, 3) uint8

        action_chunk = np.asarray(self.actions[ep_idx][t : t + self.HORIZON], dtype=np.float32)
        action = torch.from_numpy(action_chunk.reshape(-1))  # (HORIZON * ACTION_DIM,) raw, unnormalized

        z_sem_target = torch.tensor(self.z_sem[ep_idx][t + self.HORIZON], dtype=torch.float32)
        neg_indices = torch.tensor(self.neg_indices[ep_idx][t].astype(np.int64), dtype=torch.long)

        return {
            "frame": frame,
            "action": action,
            "z_sem_target": z_sem_target,
            "neg_indices": neg_indices,
            "ep_idx": ep_idx,
        }


def make_collate_fn(processor: PaliGemmaWMProcessor):
    def collate_fn(batch: list[dict]) -> dict:
        frames = [item["frame"] for item in batch]
        pixel_values = processor.image_processor(images=frames, return_tensors="pt")["pixel_values"]

        return {
            "pixel_values": pixel_values,
            "actions": torch.stack([item["action"] for item in batch]),
            "z_sem_target": torch.stack([item["z_sem_target"] for item in batch]),
            "neg_indices": torch.stack([item["neg_indices"] for item in batch]),
            "ep_idx": [item["ep_idx"] for item in batch],
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class InfoNCELoss(nn.Module):
    """Hard-negative-only InfoNCE — no in-batch negatives. Ported from dynamics_model.py."""

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        z_pred: torch.Tensor,  # (B, 768) L2-normalized
        z_pos: torch.Tensor,  # (B, 768) L2-normalized
        z_hard_neg: torch.Tensor,  # (B, 224, 768) L2-normalized
    ) -> torch.Tensor:
        B = z_pred.shape[0]
        pos_sim = (z_pred * z_pos).sum(dim=-1, keepdim=True) / self.tau  # (B, 1)
        hard_sim = torch.einsum("bd,bkd->bk", z_pred, z_hard_neg) / self.tau  # (B, 224)
        logits = torch.cat([pos_sim, hard_sim], dim=1)  # (B, 225)
        targets = torch.zeros(B, dtype=torch.long, device=z_pred.device)  # always index 0
        return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def gather_hard_negatives(dataset: PaliGemmaLangTableDataset, batch: dict, device: str) -> torch.Tensor:
    return torch.stack(
        [
            dataset.unique_zsem[ep_idx][neg_idx]
            for ep_idx, neg_idx in zip(batch["ep_idx"], batch["neg_indices"])
        ]
    ).to(device)


def save_checkpoint(model, optimizer, epoch: int, checkpoint_dir: str):
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.save_pretrained(epoch_dir)  # LoRA adapter (peft)

    base_model = model.get_base_model()
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "rand_n": base_model.rand_n.detach().cpu(),
            "projection_head_state_dict": base_model.projection_head.state_dict(),
            "action_projector_state_dict": base_model.action_projector.state_dict(),
        },
        os.path.join(epoch_dir, "extra_state.pt"),
    )
    print(f"Saved checkpoint to {epoch_dir}")


def train(
    data_dir: str,
    checkpoint_dir: str,
    epochs: int,
    batch_size: int,
    wandb_project: str,
    run_name: str | None,
    grad_accum_steps: int = 4,
    tau: float = 0.07,
):
    import wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=wandb_project,
        name=run_name,
        config=dict(
            batch_size=batch_size,
            epochs=epochs,
            grad_accum_steps=grad_accum_steps,
            tau=tau,
            checkpoint=CHECKPOINT,
        ),
    )

    processor = PaliGemmaWMProcessor.from_pretrained(CHECKPOINT)

    print(f"Loading dataset from {data_dir} ...")
    dataset = PaliGemmaLangTableDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(processor),
    )
    print(f"Dataset loaded: {len(dataset)} samples, {len(loader)} batches/epoch")
    wandb.log({"dataset/n_samples": len(dataset), "dataset/batches_per_epoch": len(loader)})

    print(f"Loading {CHECKPOINT} ...")
    model = PaliGemmaLatentDynamics.from_pretrained(CHECKPOINT, torch_dtype=torch.float32).to(device)
    # `rand_n` is a bare nn.Parameter (not an nn.Module), so `PreTrainedModel._init_weights`
    # (which only visits nn.Linear submodules) never touches it. `from_pretrained`'s
    # meta-device materialization path leaves parameters absent from the checkpoint as
    # uninitialized memory rather than re-running `__init__`'s `torch.randn(...)` — this
    # silently produces NaN (or, worse, non-NaN garbage) without an explicit reinit here.
    nn.init.normal_(model.rand_n, mean=0.0, std=0.02)
    # CHECKPOINT is base PaliGemma (never trained with actions), so action_projector has
    # no pretrained weights to load — it trains from random init alongside rand_n and
    # projection_head. action_projector.linear is an nn.Linear, so from_pretrained's
    # _init_weights *does* already reinitialize it correctly when absent from the
    # checkpoint; reinit explicitly anyway so this doesn't depend on checkpoint contents.
    init_std = getattr(model.config, "initializer_range", model.config.get_text_config().initializer_range)
    nn.init.normal_(model.action_projector.linear.weight, mean=0.0, std=init_std)
    nn.init.zeros_(model.action_projector.linear.bias)

    # `q_proj`/`v_proj` also match SigLIP's vision-tower attention (same submodule
    # names), which must stay frozen — scope the regex to `language_model.` only.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^language_model\..*\.(q_proj|v_proj)$",
    )
    model = get_peft_model(model, lora_config)

    # peft freezes everything not matched by target_modules by default — explicitly
    # unfreeze the new rand_n/projection_head heads and the pretrained action_projector.
    # The vision tower is left frozen (matched by neither target_modules nor this loop).
    for name, param in model.named_parameters():
        if "rand_n" in name or "projection_head" in name or "action_projector" in name:
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.3f}%)")

    action_proj_params = [p for n, p in model.named_parameters() if p.requires_grad and "action_projector" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "action_projector" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": 1e-4},
            {"params": action_proj_params, "lr": 1e-5},
        ]
    )

    loss_fn = InfoNCELoss(tau=tau)

    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        print(f"epoch {epoch} starting...")
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            actions = batch["actions"].to(device).view(-1, PaliGemmaLangTableDataset.HORIZON, ACTION_DIM)
            z_sem_target = batch["z_sem_target"].to(device)
            z_hard_neg = gather_hard_negatives(dataset, batch, device)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                z_pred = model(pixel_values=pixel_values, action_values=actions)
                loss = loss_fn(z_pred.float(), z_sem_target, z_hard_neg)

            (loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                B = z_pred.shape[0]
                cosine_sim_matrix = z_pred.float() @ z_pred.float().T
                mean_pairwise = (cosine_sim_matrix.sum() - B) / (B * (B - 1))

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/mean_pairwise_cosine": mean_pairwise.item(),
                    "train/lr_lora_head": optimizer.param_groups[0]["lr"],
                    "train/lr_action_proj": optimizer.param_groups[1]["lr"],
                    "step": step,
                }
            )
            epoch_loss += loss.item()
            step += 1

        if len(loader) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg = epoch_loss / len(loader)
        wandb.log({"train/epoch_loss": avg, "epoch": epoch})
        print(f"epoch {epoch}  avg_loss={avg:.4f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ashwink/full_data_5000/langtable/demos/")
    parser.add_argument(
        "--checkpoint_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "paligemma_dynamics"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--wandb_project", default="paligemma-dynamics")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
