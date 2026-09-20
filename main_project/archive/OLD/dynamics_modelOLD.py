import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langtable_dataset import LangTableDataset


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * hidden + beta


class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Action encoder: (B, 16) → (B, 256)
        self.action_encoder = nn.Sequential(
            nn.Linear(16, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        # FiLM-conditioned MLP
        self.fc1 = nn.Linear(768, 512)
        self.ln1 = nn.LayerNorm(512)
        self.film1 = FiLMLayer(256, 512)
        self.act1 = nn.Mish()
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.film2 = FiLMLayer(256, 512)
        self.act2 = nn.Mish()
        self.fc_out = nn.Linear(512, 768)
        self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, 8, 2) → (B, 16)
        global_cond = self.action_encoder(actions.flatten(1))

        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)  # (B, 768)


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
        B = z_pred.shape[0]
        pos_sim  = (z_pred * z_pos).sum(dim=-1, keepdim=True) / self.temperature   # (B, 1)
        hard_sim = torch.einsum("bd,bkd->bk", z_pred, z_hard_neg) / self.temperature  # (B, 224)
        logits   = torch.cat([pos_sim, hard_sim], dim=1)                            # (B, 225)
        targets  = torch.zeros(B, dtype=torch.long, device=z_pred.device)           # always index 0
        return F.cross_entropy(logits, targets)


def train(
    data_dir: str = "/home/ashwink/full_data_5000/langtable/demos/",
    device: str = "cuda",
    batch_size: int = 256,
    num_epochs: int = 200,
    lr: float = 3e-4,
    temperature: float = 0.1,
    wandb_project: str = "langtable-dynamics",
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
):
    import wandb

    wandb.init(
        project=wandb_project,
        config=dict(
            batch_size=batch_size,
            lr=lr,
            temperature=temperature,
            num_epochs=num_epochs,
        ),
    )

    dataset = LangTableDataset(data_dir, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} samples, {len(loader)} batches/epoch")
    wandb.log({"dataset/n_samples": len(dataset), "dataset/batches_per_epoch": len(loader)})

    model = LatentDynamicsModel().to(device)
    loss_fn = InfoNCELoss(temperature=temperature)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt.get("step", 0)
        print(f"Resumed from {resume_from} at epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, num_epochs):
        print(f"epoch {epoch} starting...")
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            B = batch["z_t"].shape[0]
            z_t = batch["z_t"].to(device)
            actions = batch["actions"].to(device)
            z_sem_target = batch["z_sem_target"].to(device)
            z_hard_neg = batch["z_hard_neg"].to(device)

            z_pred = model(z_t, actions)

            # collapse diagnostic: mean pairwise cosine similarity (approaches 1.0 → collapse)
            with torch.no_grad():
                cosine_sim_matrix = z_pred @ z_pred.T
                mean_pairwise = (cosine_sim_matrix.sum() - B) / (B * (B - 1))

            loss = loss_fn(z_pred, z_sem_target, z_hard_neg)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/mean_pairwise_cosine": mean_pairwise.item(),
                "step": step,
            })
            epoch_loss += loss.item()
            step += 1

        avg = epoch_loss / len(loader)
        wandb.log({"train/epoch_loss": avg, "epoch": epoch})
        print(f"epoch {epoch}  avg_loss={avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": avg,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run training loop with wandb")
    args = parser.parse_args()

    if args.train:
        train(resume_from=None, num_epochs=200)
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
