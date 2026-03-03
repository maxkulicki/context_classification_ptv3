"""Compare BDL (Bank Danych o Lasach) subdivision-level species records
against per-tree species labels in the LiDAR point cloud dataset.

BDL describes entire forest subdivisions (potentially tens of hectares),
while point cloud plots are ~500 m² circles, so some mismatch is expected.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Species code mapping ─────────────────────────────────────────────────────
BDL_TO_GENUS = {
    "JD": "Abies",
    "JW": "Acer", "KL": "Acer", "KL.P": "Acer",
    "OL": "Alnus", "OL.S": "Alnus",
    "BRZ": "Betula",
    "GB": "Carpinus",
    "BK": "Fagus",
    "MD": "Larix",
    "ŚW": "Picea",
    "SO": "Pinus", "SO.B": "Pinus", "SO.WE": "Pinus",
    "DB": "Quercus", "DB.S": "Quercus", "DB.B": "Quercus", "DB.C": "Quercus",
    "LP": "Tilia",
}

ALL_GENERA = sorted(set(BDL_TO_GENUS.values()))  # 11 genera

# sp_part_cd encoding → numeric proportion
PART_CD_MAP = {str(i): i * 0.10 for i in range(1, 11)}
PART_CD_MAP["MJS"] = 0.025
PART_CD_MAP["PJD"] = 0.025

# Layers considered "tree canopy"
DRZEW_LAYERS = {"DRZEW"}

BDL_PATH = Path("/home/makskulicki/tree_species_context_classification/data/plots_bdl_fused.csv")
PC_MAPPING_PATH = Path("Pointcept/data/treescanpl/sample_plotid_mapping.csv")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_bdl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    df = df.rename(columns=lambda c: c.strip())
    # Keep only rows with a species code and a proportion code
    df = df.dropna(subset=["sp_species_cd", "sp_part_cd"])
    df["sp_part_cd"] = df["sp_part_cd"].astype(str).str.strip()
    df["sp_species_cd"] = df["sp_species_cd"].astype(str).str.strip()
    df["storey_cd"] = df["storey_cd"].astype(str).str.strip()
    df["num"] = df["num"].astype(int)
    # Map species code → genus
    df["genus"] = df["sp_species_cd"].map(BDL_TO_GENUS).fillna("Other")
    # Map proportion code → numeric
    df["proportion"] = df["sp_part_cd"].map(PART_CD_MAP)
    return df


def load_pc_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genus"] = df["sample_name"].str.replace(r"_\d+$", "", regex=True)
    return df


def bdl_genus_proportions(bdl: pd.DataFrame, drzew_only: bool) -> dict[int, dict[str, float]]:
    """Return {plot_id: {genus: normalized_proportion}} from BDL."""
    if drzew_only:
        sub = bdl[bdl["storey_cd"].isin(DRZEW_LAYERS)]
    else:
        sub = bdl.copy()

    # Drop rows where proportion couldn't be mapped (shouldn't happen, but safe)
    sub = sub.dropna(subset=["proportion"])

    result = {}
    for plot_id, grp in sub.groupby("num"):
        # Aggregate by genus (sum proportions for same genus across species codes)
        genus_prop = grp.groupby("genus")["proportion"].sum()
        total = genus_prop.sum()
        if total > 0:
            genus_prop = genus_prop / total
        result[int(plot_id)] = genus_prop.to_dict()
    return result


def pc_genus_counts(mapping: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Return {plot_id: {genus: count}} from point cloud samples."""
    result = {}
    for plot_id, grp in mapping.groupby("plot_id"):
        counts = grp["genus"].value_counts().to_dict()
        result[int(plot_id)] = counts
    return result


def pc_genus_proportions(counts: dict[int, dict[str, int]]) -> dict[int, dict[str, float]]:
    """Convert counts to proportions."""
    result = {}
    for plot_id, cnts in counts.items():
        total = sum(cnts.values())
        if total > 0:
            result[plot_id] = {g: c / total for g, c in cnts.items()}
    return result


# ── Comparison functions ─────────────────────────────────────────────────────

def presence_absence_analysis(bdl_props: dict, pc_counts: dict, common_plots: list):
    """Per-genus: match (both), BDL-only, PC-only counts and per-plot Jaccard."""
    genus_match = defaultdict(int)
    genus_bdl_only = defaultdict(int)
    genus_pc_only = defaultdict(int)
    genus_bdl_total = defaultdict(int)
    genus_pc_total = defaultdict(int)
    jaccards = []

    for pid in common_plots:
        bdl_genera = {g for g in bdl_props.get(pid, {}) if g != "Other"}
        pc_genera = set(pc_counts.get(pid, {}).keys())

        intersection = bdl_genera & pc_genera
        bdl_only = bdl_genera - pc_genera
        pc_only = pc_genera - bdl_genera
        union = bdl_genera | pc_genera

        if len(union) > 0:
            jaccards.append(len(intersection) / len(union))

        for g in intersection:
            genus_match[g] += 1
        for g in bdl_only:
            genus_bdl_only[g] += 1
        for g in pc_only:
            genus_pc_only[g] += 1
        for g in bdl_genera:
            genus_bdl_total[g] += 1
        for g in pc_genera:
            genus_pc_total[g] += 1

    return genus_match, genus_bdl_only, genus_pc_only, genus_bdl_total, genus_pc_total, jaccards


def proportional_analysis(bdl_props: dict, pc_props: dict, common_plots: list):
    """Per-plot Spearman correlation and cosine similarity over genus vectors."""
    spearman_rhos = []
    cosine_sims = []
    scatter_bdl = []
    scatter_pc = []
    scatter_genus = []

    for pid in common_plots:
        bp = bdl_props.get(pid, {})
        pp = pc_props.get(pid, {})
        shared = {g for g in set(bp) | set(pp) if g != "Other"}
        if len(shared) < 2:
            continue

        bdl_vec = np.array([bp.get(g, 0.0) for g in ALL_GENERA])
        pc_vec = np.array([pp.get(g, 0.0) for g in ALL_GENERA])

        # Spearman
        rho, _ = stats.spearmanr(bdl_vec, pc_vec)
        spearman_rhos.append(rho)

        # Cosine similarity
        denom = np.linalg.norm(bdl_vec) * np.linalg.norm(pc_vec)
        if denom > 0:
            cosine_sims.append(np.dot(bdl_vec, pc_vec) / denom)

        # Scatter data
        for g in ALL_GENERA:
            bv = bp.get(g, 0.0)
            pv = pp.get(g, 0.0)
            if bv > 0 or pv > 0:
                scatter_bdl.append(bv)
                scatter_pc.append(pv)
                scatter_genus.append(g)

    return spearman_rhos, cosine_sims, scatter_bdl, scatter_pc, scatter_genus


def dominant_species_analysis(bdl_props: dict, pc_counts: dict, common_plots: list):
    """Check if BDL dominant genus matches PC dominant genus."""
    matches = 0
    total = 0
    genus_match_counts = defaultdict(int)
    genus_total_counts = defaultdict(int)

    for pid in common_plots:
        bp = {g: v for g, v in bdl_props.get(pid, {}).items() if g != "Other"}
        pc = pc_counts.get(pid, {})
        if not bp or not pc:
            continue

        bdl_dom = max(bp, key=bp.get)
        pc_dom = max(pc, key=pc.get)
        total += 1
        genus_total_counts[bdl_dom] += 1
        if bdl_dom == pc_dom:
            matches += 1
            genus_match_counts[bdl_dom] += 1

    return matches, total, genus_match_counts, genus_total_counts


# ── Plotting ─────────────────────────────────────────────────────────────────

GENUS_COLORS = {
    "Abies": "#1f77b4", "Acer": "#ff7f0e", "Alnus": "#2ca02c",
    "Betula": "#d62728", "Carpinus": "#9467bd", "Fagus": "#8c564b",
    "Larix": "#e377c2", "Picea": "#7f7f7f", "Pinus": "#bcbd22",
    "Quercus": "#17becf", "Tilia": "#aec7e8",
}


def plot_presence_absence(results_drzew, results_all, out_dir: Path):
    """Grouped bar chart: match/BDL-only/PC-only per genus, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    titles = ["DRZEW-only (tree canopy)", "All layers"]

    for ax, (match, bdl_only, pc_only, bdl_tot, pc_tot, _), title in zip(
        axes, [results_drzew, results_all], titles
    ):
        genera = ALL_GENERA
        x = np.arange(len(genera))
        w = 0.25

        match_vals = [match.get(g, 0) for g in genera]
        bdl_vals = [bdl_only.get(g, 0) for g in genera]
        pc_vals = [pc_only.get(g, 0) for g in genera]

        ax.bar(x - w, match_vals, w, label="Both (match)", color="#2ca02c")
        ax.bar(x, bdl_vals, w, label="BDL-only", color="#ff7f0e")
        ax.bar(x + w, pc_vals, w, label="PC-only", color="#1f77b4")

        ax.set_xticks(x)
        ax.set_xticklabels(genera, rotation=45, ha="right")
        ax.set_ylabel("Number of plots")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Presence / Absence: BDL vs Point Cloud", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "presence_absence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard(jaccards_drzew, jaccards_all, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, jac, title in zip(
        axes,
        [jaccards_drzew, jaccards_all],
        ["DRZEW-only", "All layers"],
    ):
        ax.hist(jac, bins=20, range=(0, 1), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(jac), color="red", linestyle="--",
                   label=f"mean={np.mean(jac):.2f}")
        ax.set_xlabel("Jaccard similarity")
        ax.set_ylabel("Number of plots")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Per-plot Jaccard Similarity (genus sets)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "jaccard_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_proportion_scatter(scatter_data_drzew, scatter_data_all, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["DRZEW-only", "All layers"]

    for ax, (s_bdl, s_pc, s_gen), title in zip(
        axes, [scatter_data_drzew, scatter_data_all], titles
    ):
        for g in ALL_GENERA:
            mask = [sg == g for sg in s_gen]
            bx = [s_bdl[i] for i, m in enumerate(mask) if m]
            py = [s_pc[i] for i, m in enumerate(mask) if m]
            if bx:
                ax.scatter(bx, py, alpha=0.4, s=20, label=g,
                           color=GENUS_COLORS.get(g, "gray"))

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("BDL proportion")
        ax.set_ylabel("PC proportion")
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    fig.suptitle("BDL vs PC Genus Proportions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "proportion_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_hist(rhos_drzew, rhos_all, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rhos, title in zip(
        axes, [rhos_drzew, rhos_all], ["DRZEW-only", "All layers"]
    ):
        ax.hist(rhos, bins=20, range=(-1, 1), edgecolor="black", alpha=0.7)
        ax.axvline(np.nanmean(rhos), color="red", linestyle="--",
                   label=f"mean={np.nanmean(rhos):.2f}")
        ax.set_xlabel("Spearman ρ")
        ax.set_ylabel("Number of plots")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Per-plot Spearman Correlation (genus proportions)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_match(dom_drzew, dom_all, out_dir: Path):
    _, _, gm_d, gt_d = dom_drzew
    _, _, gm_a, gt_a = dom_all

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    titles = ["DRZEW-only", "All layers"]

    for ax, (gm, gt), title in zip(axes, [(gm_d, gt_d), (gm_a, gt_a)], titles):
        genera = [g for g in ALL_GENERA if gt.get(g, 0) > 0]
        x = np.arange(len(genera))
        totals = [gt[g] for g in genera]
        matches = [gm.get(g, 0) for g in genera]
        rates = [m / t if t > 0 else 0 for m, t in zip(matches, totals)]

        bars = ax.bar(x, rates, color=[GENUS_COLORS.get(g, "gray") for g in genera])
        for i, (r, t) in enumerate(zip(rates, totals)):
            ax.text(i, r + 0.02, f"{r:.0%}\n(n={t})", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(genera, rotation=45, ha="right")
        ax.set_ylabel("Match rate")
        ax.set_ylim(0, 1.15)
        ax.set_title(title)

    fig.suptitle("Dominant Species Match Rate", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "dominant_match.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(
    out_dir: Path,
    n_bdl_plots, n_pc_plots, n_common,
    pa_drzew, pa_all,
    prop_drzew, prop_all,
    dom_drzew, dom_all,
):
    lines = []
    lines.append("# BDL vs Point Cloud Species Comparison Report\n")
    lines.append("## Dataset Summary\n")
    lines.append(f"- **BDL plots**: {n_bdl_plots}")
    lines.append(f"- **Point cloud plots**: {n_pc_plots}")
    lines.append(f"- **Overlapping plots**: {n_common}")
    lines.append(f"- **Genera in comparison**: {len(ALL_GENERA)} ({', '.join(ALL_GENERA)})")
    lines.append("")

    # Presence / Absence
    lines.append("## 1. Presence / Absence Agreement\n")
    lines.append("![Presence/Absence](presence_absence.png)\n")

    for label, (match, bdl_only, pc_only, bdl_tot, pc_tot, jaccards) in [
        ("DRZEW-only", pa_drzew), ("All layers", pa_all)
    ]:
        lines.append(f"### {label}\n")
        exact_match = sum(
            1 for pid in range(10**8)  # placeholder
            for _ in []
        )
        # Compute exact set match and other summary stats differently
        mean_jac = np.mean(jaccards) if jaccards else 0
        lines.append(f"- **Mean Jaccard similarity**: {mean_jac:.3f}")
        lines.append(f"- **Plots analysed**: {len(jaccards)}\n")

        lines.append("| Genus | BDL present | PC present | Both | BDL-only | PC-only | Detection rate |")
        lines.append("|-------|------------|------------|------|----------|---------|---------------|")
        for g in ALL_GENERA:
            bt = bdl_tot.get(g, 0)
            pt = pc_tot.get(g, 0)
            m = match.get(g, 0)
            bo = bdl_only.get(g, 0)
            po = pc_only.get(g, 0)
            det = f"{m/bt:.0%}" if bt > 0 else "N/A"
            lines.append(f"| {g} | {bt} | {pt} | {m} | {bo} | {po} | {det} |")
        lines.append("")

    lines.append("![Jaccard distribution](jaccard_distribution.png)\n")

    # Proportional agreement
    lines.append("## 2. Proportional Agreement\n")
    lines.append("![Proportion scatter](proportion_scatter.png)\n")

    for label, (rhos, cosines, _, _, _) in [
        ("DRZEW-only", prop_drzew), ("All layers", prop_all)
    ]:
        lines.append(f"### {label}\n")
        lines.append(f"- **Plots with 2+ shared genera**: {len(rhos)}")
        lines.append(f"- **Mean Spearman ρ**: {np.nanmean(rhos):.3f}" if rhos else "- **Mean Spearman ρ**: N/A")
        lines.append(f"- **Median Spearman ρ**: {np.nanmedian(rhos):.3f}" if rhos else "- **Median Spearman ρ**: N/A")
        lines.append(f"- **Mean cosine similarity**: {np.mean(cosines):.3f}" if cosines else "- **Mean cosine similarity**: N/A")
        lines.append("")

    lines.append("![Correlation distribution](correlation_distribution.png)\n")

    # Dominant species
    lines.append("## 3. Dominant Species Match\n")
    lines.append("![Dominant match](dominant_match.png)\n")

    for label, (matches, total, gm, gt) in [
        ("DRZEW-only", dom_drzew), ("All layers", dom_all)
    ]:
        rate = matches / total if total > 0 else 0
        lines.append(f"### {label}\n")
        lines.append(f"- **Overall match rate**: {rate:.1%} ({matches}/{total})")
        lines.append("")

    lines.append("---\n")
    lines.append("*Note: BDL describes entire forest subdivisions (potentially tens of hectares), "
                 "while point cloud plots are ~500 m² circles. Some mismatch is expected.*\n")

    report_path = out_dir / "bdl_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare BDL vs point cloud species data")
    parser.add_argument("--output_dir", type=str, default="results/bdl_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading BDL data...")
    bdl = load_bdl(BDL_PATH)
    print(f"  {len(bdl)} rows, {bdl['num'].nunique()} plots")

    print("Loading point cloud mapping...")
    pc_map = load_pc_mapping(PC_MAPPING_PATH)
    print(f"  {len(pc_map)} samples, {pc_map['plot_id'].nunique()} plots")

    # Unmapped BDL species summary
    unmapped = bdl[bdl["genus"] == "Other"]["sp_species_cd"].value_counts()
    if len(unmapped) > 0:
        print(f"\nUnmapped BDL species codes (→ 'Other', excluded from genus comparison):")
        for code, count in unmapped.items():
            print(f"  {code}: {count} rows")

    # Common plots
    bdl_plot_ids = set(bdl["num"].unique())
    pc_plot_ids = set(pc_map["plot_id"].unique())
    common_plots = sorted(bdl_plot_ids & pc_plot_ids)
    bdl_only_plots = bdl_plot_ids - pc_plot_ids
    print(f"\nPlot overlap: {len(common_plots)} common, "
          f"{len(bdl_only_plots)} BDL-only ({bdl_only_plots}), "
          f"{len(pc_plot_ids - bdl_plot_ids)} PC-only")

    # Build per-plot data
    pc_cnts = pc_genus_counts(pc_map)
    pc_props = pc_genus_proportions(pc_cnts)

    # Run analyses for both layer filters
    results = {}
    for drzew_only, label in [(True, "drzew"), (False, "all")]:
        print(f"\n{'='*60}")
        print(f"Analysis: {'DRZEW-only' if drzew_only else 'All layers'}")
        print(f"{'='*60}")

        bdl_props = bdl_genus_proportions(bdl, drzew_only=drzew_only)
        plots_with_data = [p for p in common_plots if p in bdl_props]
        print(f"  Plots with BDL data: {len(plots_with_data)}/{len(common_plots)}")

        pa = presence_absence_analysis(bdl_props, pc_cnts, plots_with_data)
        prop = proportional_analysis(bdl_props, pc_props, plots_with_data)
        dom = dominant_species_analysis(bdl_props, pc_cnts, plots_with_data)

        results[label] = {
            "bdl_props": bdl_props,
            "pa": pa,
            "prop": prop,
            "dom": dom,
        }

        print(f"  Mean Jaccard: {np.mean(pa[5]):.3f}" if pa[5] else "  No Jaccard data")
        rhos = prop[0]
        print(f"  Mean Spearman ρ: {np.nanmean(rhos):.3f} (n={len(rhos)})" if rhos else "  No correlation data")
        m, t = dom[0], dom[1]
        print(f"  Dominant match: {m}/{t} = {m/t:.1%}" if t > 0 else "  No dominant data")

    # Generate plots
    print("\nGenerating plots...")
    plot_presence_absence(results["drzew"]["pa"], results["all"]["pa"], out_dir)
    plot_jaccard(results["drzew"]["pa"][5], results["all"]["pa"][5], out_dir)
    plot_proportion_scatter(
        (results["drzew"]["prop"][2], results["drzew"]["prop"][3], results["drzew"]["prop"][4]),
        (results["all"]["prop"][2], results["all"]["prop"][3], results["all"]["prop"][4]),
        out_dir,
    )
    plot_correlation_hist(results["drzew"]["prop"][0], results["all"]["prop"][0], out_dir)
    plot_dominant_match(results["drzew"]["dom"], results["all"]["dom"], out_dir)

    # Generate report
    print("Generating report...")
    generate_report(
        out_dir,
        n_bdl_plots=len(bdl_plot_ids),
        n_pc_plots=len(pc_plot_ids),
        n_common=len(common_plots),
        pa_drzew=results["drzew"]["pa"],
        pa_all=results["all"]["pa"],
        prop_drzew=results["drzew"]["prop"],
        prop_all=results["all"]["prop"],
        dom_drzew=results["drzew"]["dom"],
        dom_all=results["all"]["dom"],
    )

    print(f"\nDone! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
