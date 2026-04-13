"""
Feature Visualization for PTv3 Tree Genus Classifier (UMAP or PCA)

Extracts features at every PTv3 encoder stage and (for context-aware models)
at every context branch output, then reduces each stage independently with
UMAP or PCA and saves plots coloured by genus and by data source.

Output files (given --output umap.png):
  Per stage:
    umap_{stage}_genus.png / _source.png / _combined.png
  Overview (all stages in one figure):
    umap_overview_genus.png
    umap_overview_source.png

Usage (run from the Pointcept root directory):

    python tools/visualize_umap.py \
        --config  configs/standardized_dataset/cls-ptv3-baseline-genus-small-10class-dual-val-8gpu.py \
        --checkpoint exp/snapshot_10class_dual_val/ptv3_small_4gpu_100ep/model/model_best_mAcc.pth \
        --data_root /net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy/ \
        --split val_id \
        --output umap.png \
        --method umap   # or --method pca
"""

import argparse
import copy
import math
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
        description="Multi-stage UMAP/PCA visualization of PTv3 features"
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
    parser.add_argument("--output", default="umap",
                        help="Output name stem (no extension needed). A subdirectory "
                             "with this name is created inside --output_dir and all "
                             "plots are saved there (default: umap)")
    parser.add_argument("--output_dir", default=None,
                        help="Root directory for output. A sub-folder named after "
                             "--output is created inside it. Defaults to "
                             "<pointcept_root>/umap_vis")
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


# ── Output path helpers ───────────────────────────────────────────────────────

def _make_out_dir(args) -> Path:
    """Return and create the output directory: <output_dir>/<stem>/"""
    root = Path(args.output_dir) if args.output_dir else POINTCEPT_ROOT / "umap_vis"
    stem = Path(args.output).stem  # strip extension if user included one
    out_dir = root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _stage_output_paths(out_dir: Path, stem: str, stage: str, suffix: str = ".png"):
    safe = stage.replace("/", "_").replace(" ", "_")
    return (
        out_dir / f"{stem}_{safe}_genus{suffix}",
        out_dir / f"{stem}_{safe}_source{suffix}",
        out_dir / f"{stem}_{safe}_combined{suffix}",
    )


def _overview_output_paths(out_dir: Path, stem: str, suffix: str = ".png"):
    return (
        out_dir / f"{stem}_overview_genus{suffix}",
        out_dir / f"{stem}_overview_source{suffix}",
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


# ── Feature extraction hooks ──────────────────────────────────────────────────

class FeatureExtractor:
    """Registers forward hooks on named submodules and accumulates features.

    stage_info is an OrderedDict mapping stage label → dict with keys:
      "module"    : nn.Module to hook
      "hook_type" : "post" (register_forward_hook) or
                    "pre"  (register_forward_pre_hook, captures input[0])
      "is_point"  : bool — True if the post-hook output is a Point object
                    that requires segment_csr mean-pooling to get [B, C].
                    Ignored for "pre" hooks and tuple outputs.
    """

    def __init__(self, stage_info: "OrderedDict[str, dict]"):
        self.stage_names = list(stage_info.keys())
        self._buffers: "dict[str, list]" = {n: [] for n in self.stage_names}
        self._handles = []

        for name, info in stage_info.items():
            module = info["module"]
            if info["hook_type"] == "pre":
                handle = module.register_forward_pre_hook(self._make_pre_hook(name))
            else:
                handle = module.register_forward_hook(
                    self._make_post_hook(name, info["is_point"])
                )
            self._handles.append(handle)

    def _make_post_hook(self, name: str, is_point: bool):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # e.g. CLSContextEncoder returns (ctx_tensor, source_tokens)
                feat = output[0].detach().cpu()
            elif is_point:
                # PTv3 encoder stage — output is a Point object
                feat = torch_scatter.segment_csr(
                    src=output.feat,
                    indptr=F.pad(output.offset, (1, 0)),
                    reduce="mean",
                ).detach().cpu()
            else:
                feat = output.detach().cpu()
            self._buffers[name].append(feat)
        return hook

    def _make_pre_hook(self, name: str):
        # Captures the first positional input to the hooked module (pre-forward).
        # Used to get the mean-pooled backbone feature entering cls_head.
        def hook(module, input):
            feat = input[0].detach().cpu()
            self._buffers[name].append(feat)
        return hook

    def collect(self) -> "dict[str, np.ndarray]":
        """Concatenate all accumulated batches → {stage: [N, C]}."""
        return {
            name: np.concatenate(bufs, axis=0)
            for name, bufs in self._buffers.items()
            if bufs  # skip stages that never fired
        }

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def get_stages(model) -> "OrderedDict[str, dict]":
    """
    Introspect model type and return an OrderedDict of hook descriptors.

    Supports:
      DefaultClassifier
      ContextFusionClassifier  (CtxCls-v1m1)
      MultiCatCtxCls           (MultiCatCtxCls-v1m1)
      MultiCatCtxClsAux        (MultiCatCtxCls-v1m2)
      MultiXAttnCtxCls         (MultiXAttnCtxCls-v1m1)
      MidFusionCtxCls          (MidFusionCtxCls-v1m1)
    """
    stages: "OrderedDict[str, dict]" = OrderedDict()
    model_type = type(model).__name__
    backbone = model.backbone

    # ── PTv3 backbone embedding ──────────────────────────────────────────────
    stages["embed"] = dict(module=backbone.embedding, hook_type="post", is_point=True)

    # ── PTv3 encoder stages (enc0, enc1, ...) — enumerated dynamically ───────
    # backbone.enc is a PointSequential whose _modules dict holds enc0, enc1, …
    for enc_name, enc_module in backbone.enc._modules.items():
        stages[enc_name] = dict(module=enc_module, hook_type="post", is_point=True)

    # ── Context branches ─────────────────────────────────────────────────────
    if model_type == "ContextFusionClassifier":
        # Single context encoder; output is tensor [B, 256]
        stages["ctx"] = dict(
            module=model.context_encoder, hook_type="post", is_point=False
        )
        # Fused feature cat(tree_feat, ctx) → [B, backbone_dim+256]
        stages["pre_cls"] = dict(
            module=model.cls_head, hook_type="pre", is_point=False
        )

    elif model_type in ("MultiCatCtxCls", "MultiCatCtxClsAux"):
        # One encoder per source; output is tensor [B, 256] each
        for i, enc in enumerate(model.context_encoders):
            key = model.context_keys[i]          # e.g. "ctx_ae"
            label = f"ctx_{key.removeprefix('ctx_')}"   # → "ctx_ae"
            stages[label] = dict(module=enc, hook_type="post", is_point=False)
        # Fused feature cat(tree_feat, ctx0, ctx1, ...) → [B, 512 + N*256]
        stages["pre_cls"] = dict(
            module=model.cls_head, hook_type="pre", is_point=False
        )

    elif model_type == "MultiXAttnCtxCls":
        # Per-source encoders
        for i, enc in enumerate(model.context_encoders):
            key = model.context_keys[i]
            label = f"ctx_{key.removeprefix('ctx_')}"
            stages[label] = dict(module=enc, hook_type="post", is_point=False)
        # norm2 output = post-cross-attention fused feature = cls_head input [B, 512]
        stages["pre_cls"] = dict(
            module=model.norm2, hook_type="post", is_point=False
        )

    elif model_type == "MidFusionCtxCls":
        # Context encoder output [B, 256]
        stages["ctx"] = dict(
            module=model.context_encoder, hook_type="post", is_point=False
        )
        # Pooled post-fusion backbone feature entering cls_head [B, backbone_embed_dim]
        # (enc3 = pre-fusion, enc4 = post-fusion backbone; this adds the pooled version)
        stages["pre_cls"] = dict(
            module=model.cls_head, hook_type="pre", is_point=False
        )

    return stages


# ── Reduction helper ──────────────────────────────────────────────────────────

def reduce_features(features: np.ndarray, method: str, args) -> np.ndarray:
    """Fit [N, C] → [N, 2] with UMAP or PCA."""
    if method == "umap":
        reducer = UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
            verbose=False,
        )
    else:
        reducer = PCA(n_components=2, random_state=args.seed)
    return reducer.fit_transform(features)


# ── Multi-stage extraction ────────────────────────────────────────────────────

def extract_features_multistage(model, dataset, batch_size, num_workers, device):
    """
    Run model(batch) with hooks active to collect per-stage features.

    Returns
    -------
    stage_features : OrderedDict[str, np.ndarray]  — {stage: [N, C]}
    genus_ids      : np.ndarray [N]
    source_ids     : np.ndarray [N]
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

    stage_info = get_stages(model)
    extractor = FeatureExtractor(stage_info)
    all_genus, all_source = [], []

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                # Run full model so all context hooks fire too.
                # Loss is computed but discarded; category is always in val batches.
                model(batch)

                all_genus.append(batch["category"].cpu().numpy().ravel())
                all_source.append(batch["source_id"].cpu().numpy().ravel())
    finally:
        extractor.remove_hooks()

    stage_features = OrderedDict(extractor.collect())
    genus_ids = np.concatenate(all_genus, axis=0)
    source_ids = np.concatenate(all_source, axis=0)
    return stage_features, genus_ids, source_ids


# ── Per-stage plots ───────────────────────────────────────────────────────────

def _scatter_kwargs(n):
    return dict(s=max(4, 14 - n // 500), alpha=0.65, linewidths=0)


def plot_by_genus(embedding, genus_ids, class_names, path, split, method,
                  stage=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    kw = _scatter_kwargs(len(genus_ids))
    m = method.upper()
    title = f"{m} — by genus ({split})"
    if stage:
        title = f"{m} [{stage}] — by genus ({split})"

    for idx, name in enumerate(class_names):
        mask = genus_ids == idx
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx / 10)],
                   label=f"{name} (n={mask.sum()})",
                   marker="o", **kw)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


def plot_by_source(embedding, source_ids, source_names, path, split, method,
                   stage=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    kw = _scatter_kwargs(len(source_ids))
    n = len(source_names)
    m = method.upper()
    title = f"{m} — by source ({split})"
    if stage:
        title = f"{m} [{stage}] — by source ({split})"

    for idx, name in enumerate(source_names):
        mask = source_ids == idx
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx / max(n - 1, 1))],
                   label=f"{name} (n={mask.sum()})",
                   marker=_MARKERS[idx % len(_MARKERS)], **kw)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


def plot_combined(embedding, genus_ids, source_ids,
                  class_names, source_names, path, split, method, stage=None):
    """Color = genus (tab10), marker shape = data source."""
    fig, ax = plt.subplots(figsize=(12, 8))
    genus_cmap = plt.get_cmap("tab10")
    kw = _scatter_kwargs(len(genus_ids))
    m = method.upper()
    title = f"{m} — color=genus, shape=source ({split})"
    if stage:
        title = f"{m} [{stage}] — color=genus, shape=source ({split})"

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

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"{m} 1"); ax.set_ylabel(f"{m} 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


# ── Overview multi-panel plots ────────────────────────────────────────────────

def _overview_grid(n_stages):
    ncols = min(n_stages, 4)
    nrows = math.ceil(n_stages / ncols)
    return nrows, ncols


def plot_overview_genus(stage_embeddings, genus_ids, class_names,
                        path, split, method):
    stages = list(stage_embeddings.keys())
    n = len(stages)
    nrows, ncols = _overview_grid(n)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.reshape(-1)
    cmap = plt.get_cmap("tab10")
    kw = _scatter_kwargs(len(genus_ids))

    for i, stage in enumerate(stages):
        ax = axes_flat[i]
        emb = stage_embeddings[stage]
        for idx, name in enumerate(class_names):
            mask = genus_ids == idx
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(idx / 10)], marker="o", **kw)
        dim = stage_embeddings[stage].shape[1]
        ax.set_title(f"{stage} ({dim}d)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    handles = [
        mlines.Line2D([], [], color=cmap(i / 10), marker="o",
                      linestyle="None", markersize=6, label=name)
        for i, name in enumerate(class_names)
        if (genus_ids == i).any()
    ]
    fig.legend(handles=handles, title="Genus",
               bbox_to_anchor=(1.0, 0.5), loc="center left",
               fontsize=8, framealpha=0.8)
    fig.suptitle(
        f"{method.upper()} — all stages, by genus ({split})", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}", flush=True)


def plot_overview_source(stage_embeddings, source_ids, source_names,
                         path, split, method):
    stages = list(stage_embeddings.keys())
    n = len(stages)
    nrows, ncols = _overview_grid(n)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.reshape(-1)
    cmap = plt.get_cmap("tab20")
    n_src = len(source_names)
    kw = _scatter_kwargs(len(source_ids))

    for i, stage in enumerate(stages):
        ax = axes_flat[i]
        emb = stage_embeddings[stage]
        for idx, name in enumerate(source_names):
            mask = source_ids == idx
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(idx / max(n_src - 1, 1))],
                       marker=_MARKERS[idx % len(_MARKERS)], **kw)
        dim = stage_embeddings[stage].shape[1]
        ax.set_title(f"{stage} ({dim}d)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    handles = [
        mlines.Line2D([], [], color=cmap(i / max(n_src - 1, 1)),
                      marker=_MARKERS[i % len(_MARKERS)],
                      linestyle="None", markersize=6, label=name)
        for i, name in enumerate(source_names)
        if (source_ids == i).any()
    ]
    fig.legend(handles=handles, title="Data source",
               bbox_to_anchor=(1.0, 0.5), loc="center left",
               fontsize=8, framealpha=0.8)
    fig.suptitle(
        f"{method.upper()} — all stages, by source ({split})", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
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
    print(f"[model] Type: {type(model).__name__}", flush=True)

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

    out_dir = _make_out_dir(args)
    stem = Path(args.output).stem
    suffix = Path(args.output).suffix or ".png"
    print(f"[output] Saving to {out_dir}/", flush=True)

    print("[features] Running multi-stage forward pass...", flush=True)
    stage_features, genus_ids, source_ids = extract_features_multistage(
        model, dataset, args.batch_size, args.num_workers, device
    )
    print(f"[features] Collected {len(stage_features)} stages:", flush=True)
    for sname, sf in stage_features.items():
        print(f"  {sname}: {sf.shape}", flush=True)

    # ── Per-stage reduction + individual plots ────────────────────────────────
    stage_embeddings: "OrderedDict[str, np.ndarray]" = OrderedDict()

    for stage_name, feats in stage_features.items():
        print(
            f"[{args.method}] Reducing '{stage_name}' {feats.shape} ...",
            flush=True,
        )
        emb = reduce_features(feats, args.method, args)
        stage_embeddings[stage_name] = emb

        pg, ps, pc = _stage_output_paths(out_dir, stem, stage_name, suffix)
        plot_by_genus(emb, genus_ids, class_names, pg, args.split, args.method,
                      stage=stage_name)
        plot_by_source(emb, source_ids, source_names, ps, args.split, args.method,
                       stage=stage_name)
        plot_combined(emb, genus_ids, source_ids, class_names, source_names,
                      pc, args.split, args.method, stage=stage_name)

    # ── Overview multi-panel plots ────────────────────────────────────────────
    ov_genus, ov_source = _overview_output_paths(out_dir, stem, suffix)
    plot_overview_genus(stage_embeddings, genus_ids, class_names,
                        ov_genus, args.split, args.method)
    plot_overview_source(stage_embeddings, source_ids, source_names,
                         ov_source, args.split, args.method)


if __name__ == "__main__":
    main()
