"""
Feature Visualization for PTv3 Tree Genus Classifier (UMAP or PCA)

Extracts 256-dimensional features from the PTv3 backbone (before the
classification head), reduces them with UMAP or PCA, and saves three plots:
  <stem>_genus.png    — colored by tree genus
  <stem>_source.png   — colored by data source / dataset
  <stem>_combined.png — color = genus, marker shape = data source

Usage (run from the Pointcept root directory):

    python tools/visualize_umap.py \
        --config configs/standardized_dataset/cls-ptv3-baseline-genus-small-10class-dual-val-8gpu.py \
        --checkpoint exp/snapshot_10class_dual_val/ptv3_small_4gpu_100ep/model/model_best_mAcc.pth \
        --data_root /net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy/ \
        --split val_id \
        --output umap.png \
        --method umap   # or --method pca
"""

import argparse
import copy
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
from umap import UMAP
from sklearn.decomposition import PCA

# Make Pointcept importable regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent
POINTCEPT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(POINTCEPT_ROOT))

from pointcept.utils.config import Config
from pointcept.utils.logger import get_root_logger
from pointcept.models import build_model
from pointcept.models.utils.structure import Point
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn


# Distinct markers for data sources (supports up to 10 sources)
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "h", "X", "d"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP visualization of PTv3 backbone features"
    )
    parser.add_argument("--config", required=True,
                        help="Path to the training config .py file")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the model checkpoint .pth file")
    parser.add_argument("--data_root", default=None,
                        help="Override the data_root from config (optional)")
    parser.add_argument("--split", default="val_id",
                        choices=["train", "val_id", "val_ood"],
                        help="Dataset split to visualize (default: val_id)")
    parser.add_argument("--output", default="umap.png",
                        help="Output base path; three files will be written "
                             "with _genus / _source / _combined suffixes "
                             "(default: umap.png)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--method", default="umap", choices=["umap", "pca"],
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cuda:N', or 'cpu'. Auto-detected if omitted.")
    return parser.parse_args()


def _output_paths(base: str):
    """Return the three output paths derived from the base path."""
    p = Path(base)
    stem, suffix = p.stem, p.suffix or ".png"
    return (
        p.with_name(f"{stem}_genus{suffix}"),
        p.with_name(f"{stem}_source{suffix}"),
        p.with_name(f"{stem}_combined{suffix}"),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path, device):
    # flash-attention is not needed for inference and avoids a hard dependency.
    # The standard attention path produces identical results.
    cfg.model.backbone.enable_flash = False

    print("[model] Building model architecture...", flush=True)
    model = build_model(cfg.model)

    print(f"[model] Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = OrderedDict()
    for key, val in ckpt["state_dict"].items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        state_dict[new_key] = val
    model.load_state_dict(state_dict, strict=True)
    print(f"[model] Loaded (epoch {ckpt['epoch']})", flush=True)

    print(f"[model] Moving to {device}...", flush=True)
    return model.to(device).eval()


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_vis_dataset(cfg, split, data_root_override):
    """Val-style transforms (no augmentation) for any split."""
    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg["split"] = split
    if data_root_override is not None:
        dataset_cfg["data_root"] = data_root_override
    return build_dataset(dataset_cfg)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, dataset, batch_size, num_workers, device):
    """
    Returns:
        features   np.ndarray [N, 256]  — mean-pooled backbone features
        genus_ids  np.ndarray [N]       — genus class index
        source_ids np.ndarray [N]       — dataset source index
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(point_collate_fn, mix_prob=0),
        pin_memory=(str(device) != "cpu"),
        drop_last=False,
    )

    all_features, all_genus, all_source = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            point = Point(batch)
            point = model.backbone(point)

            feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=F.pad(point.offset, (1, 0)),
                reduce="mean",
            )  # [B, 256]

            all_features.append(feat.cpu().numpy())
            all_genus.append(batch["category"].cpu().numpy().ravel())
            all_source.append(batch["source_id"].cpu().numpy().ravel())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_genus, axis=0),
        np.concatenate(all_source, axis=0),
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def _scatter_kwargs(n):
    """Point size and alpha scale down slightly for large sets."""
    return dict(s=max(4, 14 - n // 500), alpha=0.65, linewidths=0)


def plot_by_genus(embedding, genus_ids, class_names, path, split, method):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    kw = _scatter_kwargs(len(genus_ids))
    m = method.upper()

    for idx, name in enumerate(class_names):
        mask = genus_ids == idx
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx / 10)],
                   label=f"{name} (n={mask.sum()})",
                   marker="o", **kw)

    ax.set_title(f"{m} — by genus ({split})", fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


def plot_by_source(embedding, source_ids, source_names, path, split, method):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    kw = _scatter_kwargs(len(source_ids))
    n = len(source_names)
    m = method.upper()

    for idx, name in enumerate(source_names):
        mask = source_ids == idx
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx / max(n - 1, 1))],
                   label=f"{name} (n={mask.sum()})",
                   marker=_MARKERS[idx % len(_MARKERS)], **kw)

    ax.set_title(f"{m} — by data source ({split})", fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


def plot_combined(embedding, genus_ids, source_ids,
                  class_names, source_names, path, split, method):
    """Color = genus (tab10), marker shape = data source."""
    fig, ax = plt.subplots(figsize=(12, 8))
    genus_cmap = plt.get_cmap("tab10")
    kw = _scatter_kwargs(len(genus_ids))
    m = method.upper()

    for g_idx in range(len(class_names)):
        for s_idx in range(len(source_names)):
            mask = (genus_ids == g_idx) & (source_ids == s_idx)
            if not mask.any():
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[genus_cmap(g_idx / 10)],
                marker=_MARKERS[s_idx % len(_MARKERS)],
                **kw,
            )

    genus_handles = [
        mlines.Line2D([], [], color=genus_cmap(i / 10), marker="o",
                      linestyle="None", markersize=7, label=name)
        for i, name in enumerate(class_names)
        if (genus_ids == i).any()
    ]
    source_handles = [
        mlines.Line2D([], [], color="dimgray",
                      marker=_MARKERS[i % len(_MARKERS)],
                      linestyle="None", markersize=7, label=name)
        for i, name in enumerate(source_names)
        if (source_ids == i).any()
    ]

    leg1 = ax.legend(handles=genus_handles, title="Genus",
                     bbox_to_anchor=(1.01, 1), loc="upper left",
                     fontsize=9, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=source_handles, title="Data source",
              bbox_to_anchor=(1.01, 0), loc="lower left",
              fontsize=9, framealpha=0.8)

    ax.set_title(f"{m} — color=genus, shape=source ({split})", fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Initialize the Pointcept logger early so its messages (e.g. dataset cache
    # loading progress) are routed to stderr and visible in the terminal.
    get_root_logger()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}", flush=True)

    print(f"[config] Parsing {args.config} ...", flush=True)
    cfg = Config.fromfile(args.config)
    class_names = cfg.data.names
    print(f"[config] {len(class_names)} classes: {class_names}", flush=True)

    model = load_model(cfg, args.checkpoint, device)

    # The dataset may load a large cached .pth file (up to ~900 MB for train).
    # The Pointcept logger prints its own progress line; we add a heads-up here.
    data_root = args.data_root or cfg.data.val.get("data_root", "")
    print(
        f"[dataset] Building '{args.split}' split from {data_root} "
        f"(loading cache — may take a minute for large splits)...",
        flush=True,
    )
    dataset = build_vis_dataset(cfg, args.split, args.data_root)
    source_names = dataset.source_names
    print(
        f"[dataset] {len(dataset)} samples, "
        f"{len(source_names)} sources: {source_names}",
        flush=True,
    )

    print("[features] Running forward pass (backbone only)...", flush=True)
    features, genus_ids, source_ids = extract_features(
        model, dataset, args.batch_size, args.num_workers, device
    )
    print(
        f"[features] Done — {features.shape}  "
        f"genus: {genus_ids.shape}  source: {source_ids.shape}",
        flush=True,
    )

    if args.method == "umap":
        print(
            f"[umap] Fitting UMAP "
            f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, seed={args.seed})...",
            flush=True,
        )
        embedding = UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
            verbose=True,
        ).fit_transform(features)
    else:
        print("[pca] Fitting PCA to 2 components...", flush=True)
        embedding = PCA(n_components=2, random_state=args.seed).fit_transform(features)
    print(f"[{args.method}] Embedding: {embedding.shape}", flush=True)

    path_genus, path_source, path_combined = _output_paths(args.output)
    plot_by_genus(embedding, genus_ids, class_names, path_genus, args.split, args.method)
    plot_by_source(embedding, source_ids, source_names, path_source, args.split, args.method)
    plot_combined(embedding, genus_ids, source_ids,
                  class_names, source_names, path_combined, args.split, args.method)


if __name__ == "__main__":
    main()
