import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import open_clip


class LangTableDataset(Dataset):
    HORIZON = 8

    def __init__(self, data_dir: str, device: str | None = None, use_cached_zobs: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = data_dir
        self.device = device
        self.use_cached_zobs = use_cached_zobs

        if not use_cached_zobs:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-S13B-B90K"
            )
            self.clip_model = self.clip_model.to(device).eval()
        # CLIP not loaded when use_cached_zobs=True — task embeddings were removed,
        # so image encoding is the only remaining CLIP use.

        self.frames: list[np.ndarray] = []  # only populated when use_cached_zobs=False
        self.actions: list[np.ndarray] = []
        self.z_sem: list[np.ndarray] = []
        self.z_obs: list[np.ndarray] = []
        self.unique_zsem: list[torch.Tensor] = []
        self.neg_indices: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []

        pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))

        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            npz_path = pkl_path.replace(".pkl", "_zsem.npz")

            if not os.path.exists(npz_path):
                warnings.warn(f"Missing zsem file for {pkl_file}, skipping.")
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)

            actions = ep["actions"]                    # list or (T-1, 2) float32
            z_sem_data = np.load(npz_path)["z_sem"]   # (T, 768)
            T = len(z_sem_data)

            ep_idx = len(self.actions)

            zhard_path = pkl_path.replace(".pkl", "_zhard.npz")
            if not os.path.exists(zhard_path):
                warnings.warn(f"Missing zhard file for {pkl_file}, skipping episode.")
                continue
            zhard = np.load(zhard_path)

            if use_cached_zobs:
                zobs_path = pkl_path.replace(".pkl", "_zobs.npz")
                if not os.path.exists(zobs_path):
                    warnings.warn(f"Missing zobs file for {pkl_file}, skipping episode.")
                    continue
                self.z_obs.append(np.load(zobs_path)["z_obs"])  # (T, 768) float32

            if not use_cached_zobs:
                self.frames.append(ep["frames"])
            self.actions.append(actions)
            self.z_sem.append(z_sem_data)
            self.unique_zsem.append(torch.tensor(zhard["unique_zsem"], dtype=torch.float32))  # (N, 768)
            self.neg_indices.append(zhard["neg_indices"])  # (T, 224) int16, numpy

            for t in range(T - self.HORIZON):
                self.indices.append((ep_idx, t))

        all_actions = np.concatenate(self.actions, axis=0)  # (N_total, 2)
        self.action_stats = {
            "min": torch.tensor(all_actions.min(axis=0), dtype=torch.float32),
            "max": torch.tensor(all_actions.max(axis=0), dtype=torch.float32),
        }

    def _normalize_actions(self, actions: np.ndarray) -> torch.Tensor:
        a = torch.from_numpy(np.array(actions, dtype=np.float32))
        lo = self.action_stats["min"]
        hi = self.action_stats["max"]
        return (a - lo) / (hi - lo) * 2.0 - 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # NOTE: CLIP lives on GPU — use DataLoader with num_workers=0.
        # Pre-cache z_t as _zobs.npz files (analogous to _zsem.npz) if
        # DataLoader throughput becomes a bottleneck.
        ep_idx, t = self.indices[idx]

        if self.use_cached_zobs:
            z_t = torch.tensor(self.z_obs[ep_idx][t], dtype=torch.float32)
        else:
            frame = self.frames[ep_idx][t]  # (H, W, 3) uint8
            img = Image.fromarray(frame)
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_t = self.clip_model.encode_image(img_tensor)
                z_t = F.normalize(z_t, dim=-1).squeeze(0).cpu().float()

        actions_norm = self._normalize_actions(self.actions[ep_idx][t : t + self.HORIZON])

        z_sem_target = torch.tensor(self.z_sem[ep_idx][t + self.HORIZON], dtype=torch.float32)

        idx        = self.neg_indices[ep_idx][t].astype(np.int64)  # (224,) int64
        z_hard_neg = self.unique_zsem[ep_idx][idx]                # (224, 768)

        return {
            "z_t": z_t,
            "actions": actions_norm,
            "z_sem_target": z_sem_target,
            "z_hard_neg": z_hard_neg,
        }


if __name__ == "__main__":
    DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
    dataset = LangTableDataset(DATA_DIR)
    print(f"Total valid samples: {len(dataset)}")
    sample = dataset[0]
    for k, v in sample.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    assert sample["z_hard_neg"].shape == (224, 768), f"Expected (224, 768), got {sample['z_hard_neg'].shape}"
    assert "hard_neg_mask" not in sample, "hard_neg_mask should not be in returned dict"
    print("Assertions passed.")

