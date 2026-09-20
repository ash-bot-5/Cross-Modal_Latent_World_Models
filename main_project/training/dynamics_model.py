import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langtable_dataset import LangTableDataset, train_val_split


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * hidden + beta


class ActionEncoderConv(nn.Module):
    """Conv1d action encoder: treats the 2 action dims as channels and the 8 timesteps as
    the sequence axis, so neighboring timesteps can interact via the kernel before the
    temporal axis is collapsed by flattening (unlike a flatten-first Linear, which discards
    step order entirely)."""

    def __init__(self, action_dim: int = 2, horizon: int = 8,
                 conv_channels: int = 64, kernel_size: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(action_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.Mish()
        self.out = nn.Linear(conv_channels * horizon, out_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, horizon, action_dim) -> (B, action_dim, horizon) for Conv1d
        h = self.act(self.conv(actions.transpose(1, 2)))  # (B, conv_channels, horizon)
        return self.out(h.flatten(1))                      # (B, out_dim)


class LatentDynamicsModel(nn.Module):
    def __init__(self, horizon: int = 8):
        super().__init__()
        # Action encoder: (B, horizon, 2) → (B, 256)
        self.action_encoder = ActionEncoderConv(horizon=horizon)
        # FiLM-conditioned MLP
        self.fc1 = nn.Linear(768, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.film1 = FiLMLayer(256, 1024)
        self.act1 = nn.Mish()
        self.fc2 = nn.Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.film2 = FiLMLayer(256, 1024)
        self.act2 = nn.Mish()
        self.fc_out = nn.Linear(1024, 768)
        self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, horizon, 2)
        global_cond = self.action_encoder(actions)

        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)  # (B, 768)


def infonce_logits(
    z_pred:      torch.Tensor,  # (B, 768) L2-normalized
    z_pos:       torch.Tensor,  # (B, 768) L2-normalized
    z_hard_neg:  torch.Tensor,  # (B, N, 768) L2-normalized, all valid
    temperature: float,
) -> torch.Tensor:
    """(B, 225) logits — index 0 is the positive, 1: are hard negatives. Shared by
    InfoNCELoss.forward and ranking_metrics so both use the exact same formula."""
    pos_sim  = (z_pred * z_pos).sum(dim=-1, keepdim=True) / temperature      # (B, 1)
    hard_sim = torch.einsum("bd,bkd->bk", z_pred, z_hard_neg) / temperature  # (B, 224)
    return torch.cat([pos_sim, hard_sim], dim=1)                            # (B, 225)


def ranking_metrics(logits: torch.Tensor) -> dict:
    """Vectorized top1/top5/mean_rank over a batch of (B, 225) logits (index 0 = positive)."""
    rank = (logits > logits[:, :1]).sum(dim=1) + 1  # (B,) 1 = best
    return {
        "top1":      (rank == 1).float().mean().item(),
        "top5":      (rank <= 5).float().mean().item(),
        "mean_rank": rank.float().mean().item(),
    }


def similarity_metrics(logits: torch.Tensor, temperature: float) -> dict:
    """Raw (pre-temperature, pre-softmax) mean positive and mean hard-negative cosine
    similarity, recovered by descaling the already-computed logits (no extra matmuls)."""
    raw = logits * temperature
    return {
        "pos_sim_mean":  raw[:, 0].mean().item(),
        "hard_sim_mean": raw[:, 1:].mean().item(),
    }


def mean_pairwise_cosine(z_pred: torch.Tensor) -> torch.Tensor:
    """Batch collapse diagnostic: mean pairwise cosine similarity across z_pred (approaches 1.0 → collapse)."""
    B = z_pred.shape[0]
    cosine_sim_matrix = z_pred @ z_pred.T
    return (cosine_sim_matrix.sum() - B) / (B * (B - 1))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_pred:     torch.Tensor,  # (B, 768) L2-normalized
        z_pos:      torch.Tensor,  # (B, 768) L2-normalized
        z_hard_neg: torch.Tensor,  # (B, N, 768) L2-normalized, all valid
    ) -> torch.Tensor:
        logits  = infonce_logits(z_pred, z_pos, z_hard_neg, self.temperature)
        targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=z_pred.device)  # always index 0
        return F.cross_entropy(logits, targets)


def train(
    data_dir: str = "/home/ashwink/full_data_5000/langtable/demos/",
    device: str = "cuda",
    batch_size: int = 256,
    num_epochs: int = 50,
    lr: float = 3e-4,
    temperature: float = 0.1,
    wandb_project: str = "langtable-dynamics",
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    n_val_episodes: int = 750,
    val_seed: int = 0,
):
    import wandb

    wandb.init(
        project=wandb_project,
        config=dict(
            batch_size=batch_size,
            lr=lr,
            temperature=temperature,
            num_epochs=num_epochs,
            n_val_episodes=n_val_episodes,
            val_seed=val_seed,
        ),
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_files, val_files = train_val_split(data_dir, n_val=n_val_episodes, seed=val_seed)
    with open(os.path.join(checkpoint_dir, "train_val_split.json"), "w") as f:
        json.dump(
            {"seed": val_seed, "n_val": n_val_episodes, "train_files": train_files, "val_files": val_files},
            f, indent=2,
        )

    train_dataset = LangTableDataset(data_dir, device=device, file_list=train_files)
    val_dataset   = LangTableDataset(data_dir, device=device, file_list=val_files)
    val_dataset.action_stats = train_dataset.action_stats  # normalize val with TRAIN-only stats

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Val dataset:   {len(val_dataset)} samples, {len(val_loader)} batches/epoch")
    wandb.log({
        "dataset/n_train_samples": len(train_dataset),
        "dataset/n_val_samples": len(val_dataset),
        "dataset/train_batches_per_epoch": len(train_loader),
        "dataset/val_batches_per_epoch": len(val_loader),
    })

    model = LatentDynamicsModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt.get("step", 0)
        print(f"Resumed from {resume_from} at epoch {ckpt['epoch']}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
        for batch in train_loader:
            B = batch["z_t"].shape[0]
            z_t = batch["z_t"].to(device)
            actions = batch["actions"].to(device)
            z_sem_target = batch["z_sem_target"].to(device)
            z_hard_neg = batch["z_hard_neg"].to(device)

            z_pred = model(z_t, actions)

            # collapse diagnostic: mean pairwise cosine similarity (approaches 1.0 → collapse)
            with torch.no_grad():
                mean_pairwise = mean_pairwise_cosine(z_pred)

            logits = infonce_logits(z_pred, z_sem_target, z_hard_neg, temperature)
            targets = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                batch_metrics = ranking_metrics(logits)
                batch_sims = similarity_metrics(logits, temperature)

            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/mean_pairwise_cosine": mean_pairwise.item(),
                "train/pos_sim_mean": batch_sims["pos_sim_mean"],
                "train/hard_sim_mean": batch_sims["hard_sim_mean"],
                "step": step,
            })
            epoch_loss += loss.item()
            for k in epoch_metrics_sum:
                epoch_metrics_sum[k] += batch_metrics[k]
            step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_metrics = {k: v / len(train_loader) for k, v in epoch_metrics_sum.items()}

        # ---- Validation pass ----
        model.eval()
        val_loss_sum = 0.0
        val_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
        val_extra_sum = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                B = batch["z_t"].shape[0]
                z_t = batch["z_t"].to(device)
                actions = batch["actions"].to(device)
                z_sem_target = batch["z_sem_target"].to(device)
                z_hard_neg = batch["z_hard_neg"].to(device)

                z_pred = model(z_t, actions)
                logits = infonce_logits(z_pred, z_sem_target, z_hard_neg, temperature)
                targets = torch.zeros(B, dtype=torch.long, device=device)
                val_loss_sum += F.cross_entropy(logits, targets).item()
                batch_metrics = ranking_metrics(logits)
                for k in val_metrics_sum:
                    val_metrics_sum[k] += batch_metrics[k]

                batch_sims = similarity_metrics(logits, temperature)
                val_extra_sum["mean_pairwise_cosine"] += mean_pairwise_cosine(z_pred).item()
                val_extra_sum["pos_sim_mean"] += batch_sims["pos_sim_mean"]
                val_extra_sum["hard_sim_mean"] += batch_sims["hard_sim_mean"]
        model.train()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_extra = {k: v / len(val_loader) for k, v in val_extra_sum.items()}
        avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_top1": avg_train_metrics["top1"],
            "train/epoch_top5": avg_train_metrics["top5"],
            "train/epoch_mean_rank": avg_train_metrics["mean_rank"],
            "val/epoch_loss": avg_val_loss,
            "val/epoch_top1": avg_val_metrics["top1"],
            "val/epoch_top5": avg_val_metrics["top5"],
            "val/epoch_mean_rank": avg_val_metrics["mean_rank"],
            "val/mean_pairwise_cosine": avg_val_extra["mean_pairwise_cosine"],
            "val/pos_sim_mean": avg_val_extra["pos_sim_mean"],
            "val/hard_sim_mean": avg_val_extra["hard_sim_mean"],
        })

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
        if epoch <= 20 or epoch % 5 == 0:
            torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, os.path.join(checkpoint_dir, "best.pt"))

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run training loop with wandb")
    args = parser.parse_args()

    if args.train:
        train(resume_from=None, num_epochs=50)
    else:
        # Smoke test
        DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
        dataset = LangTableDataset(DATA_DIR)
        loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
        batch = next(iter(loader))
        for k, v in batch.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")

        model = LatentDynamicsModel()
        z_sem_pred = model(batch["z_t"], batch["actions"])
        print("z_sem_pred:", z_sem_pred.shape)

        loss_fn = InfoNCELoss(temperature=0.1)
        loss = loss_fn(
            z_sem_pred,
            batch["z_sem_target"],
            batch["z_hard_neg"],
        )
        print("loss:", loss.item())
        loss.backward()
        print("backward OK")
