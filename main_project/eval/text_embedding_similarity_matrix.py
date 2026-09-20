"""
Pairwise cosine-similarity matrix for 6 fixed statements, using epoch 8's fine-tuned CLIP
TEXT tower (checkpoints_clip_full_finetuned_wider_hl/epoch_008.pt) -- probes the geometry of
the fine-tuned text embedding space directly (no dynamics model / images involved at all).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/text_embedding_similarity_matrix.py
"""

import os
import itertools

import torch
import torch.nn.functional as F
import open_clip

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl", "epoch_008.pt"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATEMENTS = [
    "The green cube is touching the blue moon. The peg is touching the green cube",
    "The peg is touching the green cube",
    "The green cube is touching the blue moon.",
    "The green cube is not touching the blue moon. The peg is not touching the green cube",
    "The peg is not touching the green cube",
    "The green cube is not touching the blue moon.",
]


def main():
    print(f"Checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        tokens = clip_tokenizer(STATEMENTS).to(DEVICE)
        embeds = F.normalize(clip_model.encode_text(tokens), dim=-1)  # (6, 768)

    sim = (embeds @ embeds.T).cpu().numpy()  # (6, 6) cosine similarities

    labels = [f"S{i+1}" for i in range(len(STATEMENTS))]

    print("Legend:")
    for label, s in zip(labels, STATEMENTS):
        print(f"  {label}: \"{s}\"")
    print()

    header = "      " + "".join(f"{l:>8}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:>4}  " + "".join(f"{sim[i,j]:>8.4f}" for j in range(len(labels)))
        print(row)

    print("\nAll 15 unique pairs, sorted by cosine similarity (highest to lowest):")
    pairs = []
    for i, j in itertools.combinations(range(len(STATEMENTS)), 2):
        pairs.append((sim[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    for val, li, lj in pairs:
        print(f"  {li} <-> {lj}: {val:+.4f}")


if __name__ == "__main__":
    main()
