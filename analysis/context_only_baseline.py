"""
Phase 0: Context-Only Baselines (Experiments 0.0–0.4)

Trains small MLP classifiers on individual context modalities to quantify
the location prior — how well tree genus is predictable without any point cloud.

Experiments:
  0.0  majority class (floor)
  0.1  AlphaEarth (64-dim satellite embedding)
  0.2  Topo (6-dim topographic features)
  0.3  SINR (256-dim species distribution embedding)
  0.4  GeoPlantNet (24-dim species logits, log-transformed)

Train/Val/Test split:
  Test  — Wytham Woods (all) + TreeScanPL Milicz district (fold 3)
  Val   — ~15% of non-test trees, sampled at dataset level from non-TreeScanPL
  Train — everything else

Usage:
  python context_only_baseline.py --source all
  python context_only_baseline.py --source sinr --epochs 200
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sns.set_theme(style="whitegrid", font_scale=1.0)

SOURCES = ["alphaearth", "topo", "sinr", "gpn"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(df, label):
    n_before = len(df)
    df = df.drop_duplicates(subset=["dataset", "tree_id"])
    if len(df) < n_before:
        print(f"  [{label}] dropped {n_before - len(df)} duplicate (dataset, tree_id) rows")
    return df


def load_base(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "all_trees_unified.csv"))
    df = _dedup(df, "base")
    # Normalize genus: capitalise first letter to fix e.g. "pinus" → "Pinus"
    df["genus"] = df["species"].str.split().str[0].str.capitalize()
    return df


def load_topo(data_dir):
    topo_cols = ["elevation", "slope", "northness", "eastness", "tri", "tpi"]
    df = pd.read_csv(os.path.join(data_dir, "all_trees_unified_topo.csv"))
    df = _dedup(df, "topo")
    return df[["dataset", "tree_id"] + topo_cols]


def load_sinr(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "sinr_features.csv"))
    df = _dedup(df, "sinr")
    sinr_cols = [c for c in df.columns if c.startswith("sinr_")]
    return df[["dataset", "tree_id"] + sinr_cols]


def load_gpn(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "all_trees_gpn_logits.csv"), sep=";")
    df = _dedup(df, "gpn")
    meta = {"dataset", "tree_id", "species", "latitude", "longitude"}
    gpn_cols = [c for c in df.columns if c not in meta]
    return df[["dataset", "tree_id"] + gpn_cols], gpn_cols


def load_alphaearth(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "trees_alphaearth.csv"))
    df = _dedup(df, "alphaearth")
    ae_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    return df[["dataset", "tree_id"] + ae_cols], ae_cols


def load_all_data(data_dir):
    base = load_base(data_dir)
    topo = load_topo(data_dir)
    sinr = load_sinr(data_dir)
    gpn, gpn_cols = load_gpn(data_dir)
    ae, ae_cols = load_alphaearth(data_dir)

    df = base.merge(topo, on=["dataset", "tree_id"], how="left")
    df = df.merge(sinr,  on=["dataset", "tree_id"], how="left")
    df = df.merge(gpn,   on=["dataset", "tree_id"], how="left")
    df = df.merge(ae,    on=["dataset", "tree_id"], how="left")

    # Store feature column lists as module-level globals for later use
    df.attrs["gpn_cols"] = gpn_cols
    df.attrs["ae_cols"]  = ae_cols
    df.attrs["sinr_cols"] = [c for c in df.columns if c.startswith("sinr_")]
    df.attrs["topo_cols"] = ["elevation", "slope", "northness", "eastness", "tri", "tpi"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Split logic
# ─────────────────────────────────────────────────────────────────────────────

def get_milicz_plot_ids(treescanpl_dir):
    """Return set of integer plot_ids belonging to the Milicz district (fold 3)."""
    fold3_file   = os.path.join(treescanpl_dir, "treescanpl_fold3_test.txt")
    mapping_file = os.path.join(treescanpl_dir, "sample_plotid_mapping.csv")

    with open(fold3_file) as f:
        fold3_samples = {line.strip() for line in f if line.strip()}

    mapping = pd.read_csv(mapping_file)
    milicz_plots = mapping[mapping["sample_name"].isin(fold3_samples)]["plot_id"].unique()
    return set(milicz_plots.astype(int))


def make_split(df, treescanpl_dir, val_fraction=0.15, seed=42):
    """
    Assign each row to 'train', 'val', or 'test'.

    Test  — Wytham Woods (entire) + TreeScanPL Milicz district
    Val   — ~15% of remaining non-TreeScanPL trees, whole datasets at a time.
            Datasets with <200 trees stay in train.
    Train — everything else (including all non-Milicz TreeScanPL trees).
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)

    # ── Test: Wytham Woods ──
    split[df["dataset"] == "Wytham Woods"] = "test"

    # ── Test: TreeScanPL Milicz district ──
    milicz_plots = get_milicz_plot_ids(treescanpl_dir)
    is_tsc = df["dataset"] == "TreeScanPL"
    tsc_plot_ids = (
        df.loc[is_tsc, "tree_id"]
        .str.rsplit("_", n=1).str[0]
        .apply(lambda x: int(float(x)))
    )
    split[is_tsc & tsc_plot_ids.isin(milicz_plots)] = "test"

    # ── Val: from non-TreeScanPL, non-test datasets ──
    remaining_mask = (split == "train") & (df["dataset"] != "TreeScanPL")
    dataset_sizes = df[remaining_mask].groupby("dataset").size()
    eligible = dataset_sizes[dataset_sizes >= 200].index.tolist()

    eligible = list(eligible)
    rng.shuffle(eligible)

    n_target = int(remaining_mask.sum() * val_fraction)
    val_datasets = []
    accumulated = 0
    for ds in eligible:
        if accumulated >= n_target:
            break
        val_datasets.append(ds)
        accumulated += dataset_sizes[ds]

    split[remaining_mask & df["dataset"].isin(val_datasets)] = "val"
    return split, val_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df, source):
    mapping = {
        "topo":       df.attrs["topo_cols"],
        "sinr":       df.attrs["sinr_cols"],
        "gpn":        df.attrs["gpn_cols"],
        "alphaearth": df.attrs["ae_cols"],
    }
    return mapping[source]


def get_xy(df_split, source, feat_cols, label_encoder, scaler=None, fit_scaler=False):
    """Extract (X, y) arrays, drop rows with NaN features, optionally fit scaler."""
    X = df_split[feat_cols].values.astype(np.float64)

    if source == "gpn":
        # NaN = species not predicted here (outside range) → fill with 0 (no signal)
        X = np.nan_to_num(X, nan=0.0)
        X = np.log(np.clip(X, 1e-6, None))

    valid = ~np.isnan(X).any(axis=1)
    X = X[valid].astype(np.float32)

    genera = df_split["genus"].values[valid]
    # Keep only genera known to the encoder
    known_mask = np.isin(genera, label_encoder.classes_)
    X = X[known_mask]
    y = label_encoder.transform(genera[known_mask])

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    elif scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    return X, y, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ContextMLP(nn.Module):
    def __init__(self, in_dim, n_classes, source):
        super().__init__()
        if source == "topo":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, n_classes),
            )
        else:  # alphaearth, sinr, gpn — all go through 256-dim with LayerNorm
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, n_classes),
            )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def make_class_weights(y_train, n_classes, device):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, device=device)


def train_one(source, X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model     = ContextMLP(X_tr.shape[1], n_classes, source).to(device)
    weights   = make_class_weights(y_tr, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr_t  = torch.tensor(X_tr,  device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    best_macro_f1 = -1.0
    best_state    = None
    history       = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
            macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            oa       = accuracy_score(y_val, val_pred)
            history.append({"epoch": epoch, "val_macro_f1": macro_f1, "val_oa": oa})
            print(f"    epoch {epoch:3d}  val_macro_f1={macro_f1:.4f}  val_oa={oa:.4f}")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_model.pth"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, genera, device, source_name, split_name):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, device=device)).argmax(dim=1).cpu().numpy()

    oa       = accuracy_score(y, pred)
    w_f1     = f1_score(y, pred, average="weighted",  zero_division=0)
    macro_f1 = f1_score(y, pred, average="macro",     zero_division=0)
    report   = classification_report(
        y, pred, labels=range(len(genera)), target_names=genera,
        zero_division=0, output_dict=True
    )
    metrics = {
        "source":       source_name,
        "split":        split_name,
        "oa":           round(oa, 4),
        "weighted_f1":  round(w_f1, 4),
        "macro_f1":     round(macro_f1, 4),
        "n_samples":    int(len(y)),
    }
    for g in genera:
        metrics[f"f1_{g}"] = round(report.get(g, {}).get("f1-score", 0.0), 4)

    return metrics, pred


def majority_baseline(y_tr, y_val, y_te, genera):
    majority_idx  = int(np.bincount(y_tr).argmax())
    majority_name = genera[majority_idx]
    results = []
    for split_name, y in [("val", y_val), ("test", y_te)]:
        pred     = np.full(len(y), majority_idx)
        oa       = accuracy_score(y, pred)
        w_f1     = f1_score(y, pred, average="weighted",  zero_division=0)
        macro_f1 = f1_score(y, pred, average="macro",     zero_division=0)
        m = {
            "source":      "majority",
            "split":       split_name,
            "oa":          round(oa, 4),
            "weighted_f1": round(w_f1, 4),
            "macro_f1":    round(macro_f1, 4),
            "n_samples":   int(len(y)),
        }
        for g in genera:
            m[f"f1_{g}"] = 0.0
        m[f"f1_{majority_name}"] = round(
            f1_score(y == majority_idx, pred == majority_idx, zero_division=0), 4
        )
        results.append(m)
        print(f"  {split_name}: OA={oa:.4f}  macro_f1={macro_f1:.4f}  (majority={majority_name})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, genera, title, path):
    cm      = confusion_matrix(y_true, y_pred, labels=range(len(genera)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sum,
                        where=row_sum > 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(max(8, len(genera)), max(6, len(genera) - 2)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=genera, yticklabels=genera, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(all_metrics, genera, out_dir):
    test_metrics = [m for m in all_metrics if m["split"] == "test"]
    sources  = [m["source"]    for m in test_metrics]
    macro_f1 = [m["macro_f1"]  for m in test_metrics]
    oas      = [m["oa"]        for m in test_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#95a5a6"] + ["#3498db"] * (len(sources) - 1)

    for ax, vals, label in zip(axes, [macro_f1, oas], ["Macro F1", "Overall Accuracy"]):
        ax.bar(sources, vals, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by source (test set)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_metrics.png"), dpi=150)
    plt.close(fig)

    # Per-genus F1 heatmap
    genus_rows = []
    for m in test_metrics:
        row = {"source": m["source"]}
        for g in genera:
            row[g] = m.get(f"f1_{g}", 0.0)
        genus_rows.append(row)

    gdf = pd.DataFrame(genus_rows).set_index("source")
    fig, ax = plt.subplots(figsize=(max(10, len(genera) * 0.9), max(4, len(sources) * 0.8)))
    sns.heatmap(gdf, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-genus F1 by source (test set)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_genus_f1.png"), dpi=150)
    plt.close(fig)


def generate_report(all_metrics, genera, split_info, out_dir):
    lines = [
        "# Phase 0: Context-Only Baselines",
        "",
        "Predicting tree genus from location/context features alone (no point cloud).",
        "",
        "## Dataset Split",
        "",
        f"- **Train:** {split_info['n_train']} trees",
        f"- **Val:** {split_info['n_val']} trees — datasets: {split_info['val_datasets']}",
        f"- **Test:** {split_info['n_test']} trees — Wytham Woods + TreeScanPL Milicz district",
        "",
        "## Test Set Results",
        "",
        "| Source | OA | Weighted F1 | Macro F1 | N |",
        "|--------|----|-------------|----------|---|",
    ]
    for m in [m for m in all_metrics if m["split"] == "test"]:
        lines.append(f"| {m['source']} | {m['oa']:.4f} | {m['weighted_f1']:.4f} "
                     f"| {m['macro_f1']:.4f} | {m['n_samples']} |")

    lines += [
        "",
        "## Val Set Results",
        "",
        "| Source | OA | Weighted F1 | Macro F1 | N |",
        "|--------|----|-------------|----------|---|",
    ]
    for m in [m for m in all_metrics if m["split"] == "val"]:
        lines.append(f"| {m['source']} | {m['oa']:.4f} | {m['weighted_f1']:.4f} "
                     f"| {m['macro_f1']:.4f} | {m['n_samples']} |")

    lines += [
        "",
        "![Summary metrics](summary_metrics.png)",
        "",
        "![Per-genus F1](per_genus_f1.png)",
        "",
    ]

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report: {os.path.join(out_dir, 'report.md')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Context-Only Baselines")
    parser.add_argument("--source", default="all",
                        choices=["alphaearth", "topo", "sinr", "gpn", "all"],
                        help="Which context source to train (or 'all')")
    parser.add_argument("--data_dir",        default="data")
    parser.add_argument("--treescanpl_dir",  default="Pointcept/data/treescanpl")
    parser.add_argument("--results_dir",     default="results/context_only")
    parser.add_argument("--epochs",          type=int, default=200)
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    splits_dir = os.path.join(args.results_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    df = load_all_data(args.data_dir)
    df = df[df["genus"].notna()].copy()
    print(f"  Total trees with genus label: {len(df)}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")

    # ── Build splits ──
    print("\nBuilding splits...")
    df["split"], val_datasets = make_split(df, args.treescanpl_dir, seed=args.seed)
    counts = df["split"].value_counts()
    n_train = int(counts.get("train", 0))
    n_val   = int(counts.get("val",   0))
    n_test  = int(counts.get("test",  0))
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}")
    print(f"  Val datasets: {val_datasets}")
    print(f"  Test datasets: {sorted(df[df['split']=='test']['dataset'].unique())}")

    for s in ["train", "val", "test"]:
        (df[df["split"] == s]
         [["dataset", "tree_id", "species", "genus", "latitude", "longitude"]]
         .to_csv(os.path.join(splits_dir, f"{s}.csv"), index=False))

    # ── Label encoding ──
    # Fit only on training genera; val/test genera outside this set are dropped
    train_genera = sorted(df[df["split"] == "train"]["genus"].unique())
    le = LabelEncoder()
    le.fit(train_genera)
    genera = [str(g) for g in le.classes_]  # plain str, not np.str_
    n_classes = len(genera)
    print(f"\n  Genera ({n_classes}): {genera}")

    df_train = df[df["split"] == "train"]
    df_val   = df[df["split"] == "val"]
    df_test  = df[df["split"] == "test"]

    # ── Exp 0.0: Majority class ──
    print("\n" + "="*60)
    print("Exp 0.0: Majority class")
    print("="*60)
    y_tr_all  = le.transform(df_train["genus"].values)
    y_val_all = le.transform(df_val[df_val["genus"].isin(genera)]["genus"].values)
    y_te_all  = le.transform(df_test[df_test["genus"].isin(genera)]["genus"].values)

    all_metrics = majority_baseline(y_tr_all, y_val_all, y_te_all, genera)

    sources_to_run = SOURCES if args.source == "all" else [args.source]

    for source in sources_to_run:
        print("\n" + "="*60)
        print(f"Exp: {source}")
        print("="*60)

        out_dir    = os.path.join(args.results_dir, source)
        feat_cols  = get_feature_cols(df, source)

        # Filter to rows that have this source's features (non-NaN in first col)
        df_tr_s  = df_train[df_train[feat_cols[0]].notna()].copy()
        df_val_s = df_val[df_val[feat_cols[0]].notna()].copy()
        df_te_s  = df_test[df_test[feat_cols[0]].notna()].copy()

        print(f"  Rows with features — Train: {len(df_tr_s)}  Val: {len(df_val_s)}  Test: {len(df_te_s)}")

        if len(df_tr_s) == 0 or len(df_val_s) == 0 or len(df_te_s) == 0:
            print("  Skipping — insufficient data in one split.")
            continue

        X_tr,  y_tr,  scaler = get_xy(df_tr_s,  source, feat_cols, le, fit_scaler=True)
        X_val, y_val, _      = get_xy(df_val_s, source, feat_cols, le, scaler=scaler)
        X_te,  y_te,  _      = get_xy(df_te_s,  source, feat_cols, le, scaler=scaler)

        print(f"  Feature dim: {X_tr.shape[1]}  Train samples: {len(X_tr)}")

        model = train_one(
            source, X_tr, y_tr, X_val, y_val,
            n_classes, args.epochs, args.batch_size, device, out_dir,
        )

        source_metrics = []
        for split_name, X, y in [("val", X_val, y_val), ("test", X_te, y_te)]:
            m, pred = evaluate(model, X, y, genera, device, source, split_name)
            source_metrics.append(m)
            all_metrics.append(m)
            print(f"  {split_name}: OA={m['oa']:.4f}  w_f1={m['weighted_f1']:.4f}  macro_f1={m['macro_f1']:.4f}")

            plot_confusion_matrix(
                y, pred, genera,
                f"{source} — {split_name}  macro_f1={m['macro_f1']:.3f}",
                os.path.join(out_dir, f"confusion_{split_name}.png"),
            )

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(source_metrics, f, indent=2)

    # ── Summary ──
    print("\n" + "="*60)
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(args.results_dir, "summary_table.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    print(summary_df[["source", "split", "oa", "weighted_f1", "macro_f1"]]
          .sort_values(["split", "macro_f1"], ascending=[True, False])
          .to_string(index=False))

    if args.source == "all":
        split_info = {
            "n_train":      n_train,
            "n_val":        n_val,
            "n_test":       n_test,
            "val_datasets": ", ".join(val_datasets),
        }
        plot_summary(all_metrics, genera, args.results_dir)
        generate_report(all_metrics, genera, split_info, args.results_dir)


if __name__ == "__main__":
    main()
