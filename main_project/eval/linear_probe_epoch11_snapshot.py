"""
Single-epoch linear-probe snapshot for epoch 11 of
training/checkpoints_clip_full_fine-tuned_with_subopt_data (the "expert+subopt-model"
checkpoint dir also swept by linear_probe_MLP_output.py).

Unlike linear_probe_MLP_output.py (which sweeps every epoch across two checkpoint dirs
and re-encodes frames through each checkpoint's own fine-tuned CLIP tower), this script
needs no CLIP/model forward pass at all: z_pred for epoch 11 is already cached at
linear_probe_MLP_output_charts/zpred_cache/expert+subopt-model_epoch011.npz, row-aligned
with the checkpoint-independent ground-truth cache (ground_truth_data.npz). This is a
pure sklearn-on-cached-arrays job.

Three linear probes, fit on z_pred, evaluated on the VALIDATION split only:
  1. ab_touch        classification (LogisticRegression) -> val_auc
     "is start_block (A) touching oracle_target_block (B) at t+8" -- how linearly
     separable is A-touching-B from the z_pred latent.
  2. peg_touch_start classification (LogisticRegression) -> val_auc
     "is the peg touching start_block (A) at t+8" -- how linearly separable is
     peg-touching-A from the z_pred latent.
  3. dist_ab         regression (LinearRegression) -> pearson_r
     linear probe predicts dist(A, B) at t+8 from z_pred; pearson_r is the correlation
     between the probe's prediction and the ground-truth distance on held-out rows.

fit_classification_probe / fit_regression_probe are inline-copied verbatim from
linear_probe_MLP_output.py (lines ~348-428) rather than imported, so this script stays
free of the open_clip/torch imports that module pulls in at load time for its (unused
here) CLIP re-encoding path.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/linear_probe_epoch11_snapshot.py
"""

import os
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCH = 11
CKPT_TAG = "expert+subopt-model"

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(MAIN_PROJECT_DIR, "linear_probe_MLP_output_charts")
GT_CACHE_PATH = os.path.join(CHARTS_DIR, "ground_truth_data.npz")
ZPRED_CACHE_PATH = os.path.join(CHARTS_DIR, "zpred_cache", f"{CKPT_TAG}_epoch{EPOCH:03d}.npz")
OUT_DIR = os.path.join(CHARTS_DIR, "epoch11_snapshot")

# dataviz skill reference palette (references/palette.md) -- light chart surface.
COLOR_SURFACE = "#fcfcfb"
COLOR_SERIES_1 = "#2a78d6"   # categorical slot 1, blue
COLOR_SERIES_2 = "#eb6834"   # categorical slot 2, orange
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_BASELINE = "#c3c2b7"


# ---------------------------------------------------------------------------
# Probe fitting -- inline-copied verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

def fit_classification_probe(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                              val_mask: np.ndarray, meta: dict) -> dict:
    result = dict(meta)
    result["probe_type"] = "classification"
    result["n_train"] = int(train_mask.sum())
    result["n_val"] = int(val_mask.sum())

    if train_mask.sum() == 0 or val_mask.sum() == 0 or np.unique(y[train_mask]).size < 2:
        result["skipped"] = True
        return result

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    result["n_pos_train"] = int(y_train.sum())
    result["n_neg_train"] = int((~y_train).sum())
    result["n_pos_val"] = int(y_val.sum())
    result["n_neg_val"] = int((~y_val).sum())
    result["majority_baseline_acc"] = float(max(y_val.mean(), 1 - y_val.mean()))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    result["train_acc"] = float(clf.score(X_train, y_train))
    result["val_acc"] = float(clf.score(X_val, y_val))

    pred_val = clf.predict(X_val)
    result["precision"] = float(precision_score(y_val, pred_val, zero_division=0))
    result["recall"] = float(recall_score(y_val, pred_val, zero_division=0))
    result["f1"] = float(f1_score(y_val, pred_val, zero_division=0))

    if np.unique(y_val).size < 2:
        result["val_auc"] = None
        result["auc_undefined"] = True
    else:
        margins = clf.decision_function(X_val)
        result["val_auc"] = float(roc_auc_score(y_val, margins))
        result["auc_undefined"] = False
        result["val_margins"] = margins
        result["val_labels"] = y_val

    result["skipped"] = False
    return result


def fit_regression_probe(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                          val_mask: np.ndarray, meta: dict) -> dict:
    result = dict(meta)
    result["probe_type"] = "regression"
    result["n_train"] = int(train_mask.sum())
    result["n_val"] = int(val_mask.sum())

    if train_mask.sum() < 2 or val_mask.sum() < 2:
        result["skipped"] = True
        return result

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    result["label_mean_val"] = float(y_val.mean())
    result["label_std_val"] = float(y_val.std())

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    result["train_r2"] = float(reg.score(X_train, y_train))
    result["val_r2"] = float(reg.score(X_val, y_val))

    pred_val = reg.predict(X_val)
    result["val_mae"] = float(mean_absolute_error(y_val, pred_val))
    result["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, pred_val)))

    if np.std(pred_val) > 1e-8 and np.std(y_val) > 1e-8:
        r, _ = pearsonr(y_val, pred_val)
        result["pearson_r"] = float(r)
    else:
        result["pearson_r"] = None

    mean_baseline_pred = np.full_like(y_val, y_train.mean())
    result["mean_baseline_rmse"] = float(np.sqrt(mean_squared_error(y_val, mean_baseline_pred)))

    result["val_pred"] = pred_val
    result["val_true"] = y_val

    result["skipped"] = False
    return result


# ---------------------------------------------------------------------------
# Plotting -- dataviz-skill styling
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BASELINE)
    ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_MUTED)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)


def plot_score_histogram_by_label(margins: np.ndarray, labels: np.ndarray, val_auc: float,
                                   pos_name: str, neg_name: str, title: str, annotation: str,
                                   out_path: str, n_bins: int = 30):
    """Histogram of the logistic probe's validation decision-function score, split by
    true label -- blue for the negative class, orange for the positive class -- with a
    dotted vertical line at x=0, the classifier's actual decision boundary."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    neg_scores = margins[~labels]
    pos_scores = margins[labels]

    lo, hi = float(margins.min()), float(margins.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    bins = np.linspace(lo - pad, hi + pad, n_bins)

    ax.hist(neg_scores, bins=bins, color=COLOR_SERIES_1, alpha=0.75, zorder=3,
            label=f"{neg_name} (n={len(neg_scores)})")
    ax.hist(pos_scores, bins=bins, color=COLOR_SERIES_2, alpha=0.75, zorder=3,
            label=f"{pos_name} (n={len(pos_scores)})")

    ax.axvline(0.0, color=COLOR_TEXT_PRIMARY, linestyle=":", linewidth=1.75, zorder=4,
               label="decision boundary")

    ax.set_xlabel("logistic probe decision-function score (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("count", color=COLOR_TEXT_SECONDARY)
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_SECONDARY)

    _style_axes(ax)

    full_annotation = f"val_auc = {val_auc:.3f}" + (f"\n{annotation}" if annotation else "")
    ax.text(0.98, 0.98, full_annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=COLOR_TEXT_PRIMARY,
            bbox=dict(boxstyle="round", facecolor=COLOR_SURFACE, edgecolor=COLOR_BASELINE, alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, pearson_r: float,
                             val_r2: float, title: str, annotation: str, out_path: str):
    """Scatter of the linear probe's predicted dist(A,B) vs the true dist(A,B) on
    validation rows, with a dotted y=x reference line (perfect prediction)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.scatter(y_true, y_pred, s=14, color=COLOR_SERIES_1, alpha=0.5,
               edgecolors="none", zorder=3)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLOR_TEXT_PRIMARY,
            linestyle=":", linewidth=1.75, zorder=4, label="y = x (perfect prediction)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("true dist(A, B) at t+8 (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("linear probe's predicted dist(A, B) (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_SECONDARY)

    _style_axes(ax)
    ax.set_aspect("equal", adjustable="box")

    full_annotation = f"pearson_r = {pearson_r:.3f}\nval_r2 = {val_r2:.3f}" + (f"\n{annotation}" if annotation else "")
    ax.text(0.98, 0.02, full_annotation, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=COLOR_TEXT_PRIMARY,
            bbox=dict(boxstyle="round", facecolor=COLOR_SURFACE, edgecolor=COLOR_BASELINE, alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading ground-truth cache: {GT_CACHE_PATH}")
    gt = np.load(GT_CACHE_PATH, allow_pickle=False)

    print(f"Loading z_pred cache: {ZPRED_CACHE_PATH}")
    z_pred = np.load(ZPRED_CACHE_PATH)["z_pred"]

    n_gt = len(gt["is_val"])
    assert z_pred.shape[0] == n_gt, (
        f"row mismatch: z_pred has {z_pred.shape[0]} rows, ground truth has {n_gt}"
    )
    print(f"Rows: {n_gt}  (z_pred dim={z_pred.shape[1]})")

    is_val = gt["is_val"].astype(bool)
    train_mask = ~is_val
    val_mask = is_val
    print(f"train={train_mask.sum()}  val={val_mask.sum()}\n")

    base_meta = {"checkpoint_dir_tag": CKPT_TAG, "epoch": EPOCH}
    results = {}

    print("Fitting ab_touch (A-touching-B) classification probe ...")
    r_ab = fit_classification_probe(
        z_pred, gt["ab_touch"].astype(bool), train_mask, val_mask,
        {**base_meta, "probe": "ab_touch"},
    )
    results["ab_touch"] = r_ab
    print(f"  val_auc = {r_ab.get('val_auc')}  (n_pos_val={r_ab.get('n_pos_val')}, n_neg_val={r_ab.get('n_neg_val')})")

    print("Fitting peg_touch_start (peg-touching-A) classification probe ...")
    r_peg = fit_classification_probe(
        z_pred, gt["peg_touch_start"].astype(bool), train_mask, val_mask,
        {**base_meta, "probe": "peg_touch_start"},
    )
    results["peg_touch_start"] = r_peg
    print(f"  val_auc = {r_peg.get('val_auc')}  (n_pos_val={r_peg.get('n_pos_val')}, n_neg_val={r_peg.get('n_neg_val')})")

    print("Fitting dist_ab (dist(A,B)) regression probe ...")
    r_dist = fit_regression_probe(
        z_pred, gt["dist_ab"].astype(np.float32), train_mask, val_mask,
        {**base_meta, "probe": "dist_ab"},
    )
    results["dist_ab"] = r_dist
    print(f"  val_r2 = {r_dist.get('val_r2')}  pearson_r = {r_dist.get('pearson_r')}")

    ARRAY_KEYS = ("val_margins", "val_labels", "val_pred", "val_true")
    results_json = {
        probe_name: {k: v for k, v in r.items() if k not in ARRAY_KEYS}
        for probe_name, r in results.items()
    }
    results_path = os.path.join(OUT_DIR, "results_epoch11.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results -> {results_path}")

    # -- Charts --
    if not r_ab.get("skipped") and r_ab.get("val_auc") is not None:
        plot_score_histogram_by_label(
            r_ab["val_margins"], r_ab["val_labels"], r_ab["val_auc"],
            pos_name="A touching B", neg_name="A not touching B",
            title="A-touching-B: linear separability from z_pred",
            annotation=f"epoch {EPOCH} ({CKPT_TAG})  n_val={r_ab['n_val']}",
            out_path=os.path.join(OUT_DIR, "ab_touch_val_auc.png"),
        )
        print("Saved ab_touch_val_auc.png")
    else:
        print("Skipped ab_touch chart (probe was skipped or AUC undefined)")

    if not r_peg.get("skipped") and r_peg.get("val_auc") is not None:
        plot_score_histogram_by_label(
            r_peg["val_margins"], r_peg["val_labels"], r_peg["val_auc"],
            pos_name="peg touching A", neg_name="peg not touching A",
            title="peg-touching-A: linear separability from z_pred",
            annotation=f"epoch {EPOCH} ({CKPT_TAG})  n_val={r_peg['n_val']}",
            out_path=os.path.join(OUT_DIR, "peg_touch_start_val_auc.png"),
        )
        print("Saved peg_touch_start_val_auc.png")
    else:
        print("Skipped peg_touch_start chart (probe was skipped or AUC undefined)")

    if not r_dist.get("skipped") and r_dist.get("pearson_r") is not None:
        plot_regression_scatter(
            r_dist["val_true"], r_dist["val_pred"], r_dist["pearson_r"], r_dist["val_r2"],
            title="dist(A,B): linear probe correlation with ground truth",
            annotation=f"epoch {EPOCH} ({CKPT_TAG})  n_val={r_dist['n_val']}",
            out_path=os.path.join(OUT_DIR, "dist_ab_val_pearson_r.png"),
        )
        print("Saved dist_ab_val_pearson_r.png")
    else:
        print("Skipped dist_ab chart (probe was skipped or pearson_r undefined)")

    print(f"\nDone. Outputs saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
