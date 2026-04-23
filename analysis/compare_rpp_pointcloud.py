"""Compare RPP (species presence probability map) against per-tree species
labels in the LiDAR point cloud dataset.

RPP gives an independent probability of each species being present at a
location. PC gives binary ground truth (genus present or absent per plot).
We evaluate RPP as a binary classifier using AUC and probability separation.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Species mapping: RPP column → genus ──────────────────────────────────────
RPP_TO_GENUS = {
    "Abies_alba": "Abies",
    "Acer_pseudoplatanus": "Acer",
    "Alnus_glutinosa": "Alnus",
    "Alnus_incana": "Alnus",
    "Betula_sp": "Betula",
    "Carpinus_betulus": "Carpinus",
    "Fagus_sylvatica": "Fagus",
    "Larix_decidua": "Larix",
    "Picea_abies": "Picea",
    "Pinus_sylvestris": "Pinus",
    "Quercus_robur": "Quercus",
    "Tilia_sp": "Tilia",
}

# RPP columns not in our dataset (ignored)
RPP_UNMAPPED = [
    "Fraxinus_excelsior", "Populus_tremula", "Prunus_avium",
    "Salix_caprea", "Sorbus_aucuparia",
]

ALL_GENERA = sorted(set(RPP_TO_GENUS.values()))

RPP_PATH = Path("/home/makskulicki/tree_species_context_classification/data/plots_rpp_probabilities.csv")
PC_MAPPING_PATH = Path("Pointcept/data/treescanpl/sample_plotid_mapping.csv")

GENUS_COLORS = {
    "Abies": "#1f77b4", "Acer": "#ff7f0e", "Alnus": "#2ca02c",
    "Betula": "#d62728", "Carpinus": "#9467bd", "Fagus": "#8c564b",
    "Larix": "#e377c2", "Picea": "#7f7f7f", "Pinus": "#bcbd22",
    "Quercus": "#17becf", "Tilia": "#aec7e8",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rpp(path: Path) -> pd.DataFrame:
    """Load RPP probabilities and aggregate to genus level (max of species)."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    df = df.rename(columns=lambda c: c.strip())
    df["num"] = df["num"].astype(int)

    # Aggregate species → genus (max probability for genera with multiple species)
    for genus in ALL_GENERA:
        cols = [c for c, g in RPP_TO_GENUS.items() if g == genus and c in df.columns]
        df[genus] = df[cols].max(axis=1)

    return df[["num"] + ALL_GENERA].copy()


def load_pc_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genus"] = df["sample_name"].str.replace(r"_\d+$", "", regex=True)
    return df


def pc_genus_presence(mapping: pd.DataFrame) -> dict[int, set[str]]:
    """Return {plot_id: set of genera present} from point cloud."""
    result = {}
    for plot_id, grp in mapping.groupby("plot_id"):
        result[int(plot_id)] = set(grp["genus"].unique())
    return result


# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_auc_and_probs(rpp: pd.DataFrame, pc_presence: dict, common_plots: list):
    """For each genus, compute AUC and collect probabilities split by presence."""
    aucs = {}
    probs_present = {}    # genus → list of RPP probs where PC has it
    probs_absent = {}     # genus → list of RPP probs where PC doesn't

    rpp_indexed = rpp.set_index("num")

    for genus in ALL_GENERA:
        labels = []
        scores = []
        present_list = []
        absent_list = []

        for pid in common_plots:
            prob = rpp_indexed.loc[pid, genus]
            is_present = genus in pc_presence.get(pid, set())
            labels.append(int(is_present))
            scores.append(prob)
            if is_present:
                present_list.append(prob)
            else:
                absent_list.append(prob)

        probs_present[genus] = present_list
        probs_absent[genus] = absent_list

        # AUC requires both classes
        if sum(labels) > 0 and sum(labels) < len(labels):
            aucs[genus] = roc_auc_score(labels, scores)
        else:
            aucs[genus] = None

    return aucs, probs_present, probs_absent


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_auc(aucs: dict, out_dir: Path):
    """Bar chart of per-genus AUC."""
    genera = [g for g in ALL_GENERA if aucs.get(g) is not None]
    auc_vals = [aucs[g] for g in genera]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(genera))
    bars = ax.bar(x, auc_vals, color=[GENUS_COLORS[g] for g in genera])

    for i, v in enumerate(auc_vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genera, rotation=45, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="random (0.5)")
    ax.set_title("RPP Presence Probability vs Point Cloud Labels: AUC per Genus\n"
                 "(higher = RPP better distinguishes plots with/without the genus)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "rpp_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prob_boxplots(probs_present: dict, probs_absent: dict, out_dir: Path):
    """Side-by-side box plots of RPP probability for present vs absent."""
    genera = [g for g in ALL_GENERA
              if len(probs_present.get(g, [])) > 0 and len(probs_absent.get(g, [])) > 0]

    fig, ax = plt.subplots(figsize=(12, 6))
    positions_absent = []
    positions_present = []
    data_absent = []
    data_present = []
    tick_positions = []
    tick_labels = []

    spacing = 3
    for i, g in enumerate(genera):
        base = i * spacing
        positions_absent.append(base)
        positions_present.append(base + 1)
        data_absent.append(probs_absent[g])
        data_present.append(probs_present[g])
        tick_positions.append(base + 0.5)
        tick_labels.append(g)

    bp_abs = ax.boxplot(data_absent, positions=positions_absent, widths=0.7,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
    bp_pres = ax.boxplot(data_present, positions=positions_present, widths=0.7,
                         patch_artist=True, showfliers=True,
                         flierprops=dict(marker=".", markersize=3, alpha=0.5))

    for patch in bp_abs["boxes"]:
        patch.set_facecolor("#d9d9d9")
    for patch in bp_pres["boxes"]:
        patch.set_facecolor("#2ca02c")

    # Add sample size annotations
    for i, g in enumerate(genera):
        base = i * spacing
        n_abs = len(probs_absent[g])
        n_pres = len(probs_present[g])
        y_top = max(max(probs_absent[g]), max(probs_present[g]))
        ax.text(base, -0.06, f"n={n_abs}", ha="center", fontsize=7, color="gray")
        ax.text(base + 1, -0.06, f"n={n_pres}", ha="center", fontsize=7, color="#2ca02c")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("RPP probability")
    ax.set_title("RPP Probability Distribution: Genus Absent vs Present in Point Cloud")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d9d9d9", edgecolor="black", label="Absent in PC"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Present in PC"),
    ])

    fig.tight_layout()
    fig.savefig(out_dir / "rpp_prob_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(out_dir: Path, n_rpp, n_pc, n_common, aucs, probs_present, probs_absent):
    lines = []
    lines.append("# RPP vs Point Cloud Species Comparison\n")
    lines.append("**Question**: How well do RPP presence probabilities predict "
                 "which genera actually occur in our point cloud plots?\n")
    lines.append("RPP provides independent probability estimates of each species being present "
                 "at a location. We treat PC genus presence/absence as ground truth and "
                 "evaluate RPP as a binary classifier using AUC.\n")

    lines.append("## Dataset\n")
    lines.append(f"- **RPP plots**: {n_rpp}")
    lines.append(f"- **PC plots**: {n_pc}")
    lines.append(f"- **Overlapping**: {n_common}")
    lines.append(f"- **Genera**: {len(ALL_GENERA)} ({', '.join(ALL_GENERA)})")
    lines.append(f"- **Unmapped RPP species** (not in PC dataset): {', '.join(RPP_UNMAPPED)}")
    lines.append("")

    # AUC table
    lines.append("## 1. Per-Genus AUC\n")
    lines.append("![AUC](rpp_auc.png)\n")
    lines.append("| Genus | AUC | PC present | PC absent | Median prob (present) | Median prob (absent) |")
    lines.append("|-------|-----|-----------|-----------|----------------------|---------------------|")
    for g in ALL_GENERA:
        auc = aucs.get(g)
        auc_str = f"{auc:.3f}" if auc is not None else "N/A"
        n_pres = len(probs_present.get(g, []))
        n_abs = len(probs_absent.get(g, []))
        med_pres = f"{np.median(probs_present[g]):.3f}" if probs_present.get(g) else "N/A"
        med_abs = f"{np.median(probs_absent[g]):.3f}" if probs_absent.get(g) else "N/A"
        lines.append(f"| {g} | {auc_str} | {n_pres} | {n_abs} | {med_pres} | {med_abs} |")
    lines.append("")

    # Box plots
    lines.append("## 2. Probability Distributions\n")
    lines.append("![Box plots](rpp_prob_boxplots.png)\n")

    mean_auc = np.mean([v for v in aucs.values() if v is not None])
    lines.append(f"**Mean AUC across genera**: {mean_auc:.3f}\n")

    lines.append("---\n")
    lines.append("*AUC = 1.0 means RPP perfectly separates present/absent; "
                 "0.5 means no better than random.*\n")

    (out_dir / "rpp_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare RPP vs point cloud species data")
    parser.add_argument("--output_dir", type=str, default="results/rpp_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    rpp = load_rpp(RPP_PATH)
    pc_map = load_pc_mapping(PC_MAPPING_PATH)
    print(f"  RPP: {len(rpp)} plots")
    print(f"  PC:  {len(pc_map)} samples, {pc_map['plot_id'].nunique()} plots")

    pc_pres = pc_genus_presence(pc_map)

    rpp_plot_ids = set(rpp["num"].unique())
    pc_plot_ids = set(pc_map["plot_id"].unique())
    common_plots = sorted(rpp_plot_ids & pc_plot_ids)
    print(f"\nPlots: {len(common_plots)} common, "
          f"{len(rpp_plot_ids - pc_plot_ids)} RPP-only, "
          f"{len(pc_plot_ids - rpp_plot_ids)} PC-only")

    print("\nComputing AUC per genus...")
    aucs, probs_present, probs_absent = compute_auc_and_probs(rpp, pc_pres, common_plots)
    for g in ALL_GENERA:
        auc = aucs.get(g)
        auc_str = f"{auc:.3f}" if auc is not None else "N/A"
        print(f"  {g:12s}  AUC={auc_str}  (present={len(probs_present[g])}, absent={len(probs_absent[g])})")

    mean_auc = np.mean([v for v in aucs.values() if v is not None])
    print(f"\n  Mean AUC: {mean_auc:.3f}")

    print("\nGenerating plots...")
    plot_auc(aucs, out_dir)
    plot_prob_boxplots(probs_present, probs_absent, out_dir)

    print("Generating report...")
    generate_report(out_dir, len(rpp_plot_ids), len(pc_plot_ids), len(common_plots),
                    aucs, probs_present, probs_absent)

    print(f"Done! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
