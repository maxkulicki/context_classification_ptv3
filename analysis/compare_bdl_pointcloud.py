"""Compare BDL (Bank Danych o Lasach) subdivision-level species records
against per-tree species labels in the LiDAR point cloud dataset.

BDL describes entire forest subdivisions (potentially tens of hectares),
while point cloud plots are ~500 m² circles, so some mismatch is expected.

Focus: how well does BDL cover the species actually present in our plots?
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

ALL_GENERA = sorted(set(BDL_TO_GENUS.values()))

BDL_PATH = Path("/home/makskulicki/tree_species_context_classification/data/plots_bdl_fused.csv")
PC_MAPPING_PATH = Path("Pointcept/data/treescanpl/sample_plotid_mapping.csv")

GENUS_COLORS = {
    "Abies": "#1f77b4", "Acer": "#ff7f0e", "Alnus": "#2ca02c",
    "Betula": "#d62728", "Carpinus": "#9467bd", "Fagus": "#8c564b",
    "Larix": "#e377c2", "Picea": "#7f7f7f", "Pinus": "#bcbd22",
    "Quercus": "#17becf", "Tilia": "#aec7e8",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_bdl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    df = df.rename(columns=lambda c: c.strip())
    df = df.dropna(subset=["sp_species_cd"])
    df["sp_species_cd"] = df["sp_species_cd"].astype(str).str.strip()
    df["num"] = df["num"].astype(int)
    df["genus"] = df["sp_species_cd"].map(BDL_TO_GENUS).fillna("Other")
    return df


def load_pc_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genus"] = df["sample_name"].str.replace(r"_\d+$", "", regex=True)
    return df


def bdl_genus_sets(bdl: pd.DataFrame) -> dict[int, set[str]]:
    """Return {plot_id: set of genera} from BDL (excluding 'Other')."""
    result = {}
    for plot_id, grp in bdl.groupby("num"):
        genera = set(grp["genus"].unique()) - {"Other"}
        if genera:
            result[int(plot_id)] = genera
    return result


def pc_genus_counts(mapping: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Return {plot_id: {genus: count}} from point cloud samples."""
    result = {}
    for plot_id, grp in mapping.groupby("plot_id"):
        result[int(plot_id)] = grp["genus"].value_counts().to_dict()
    return result


# ── Analysis ─────────────────────────────────────────────────────────────────

def presence_absence_analysis(bdl_sets: dict, pc_counts: dict, common_plots: list):
    """Per-genus: match (both), BDL-only, PC-only plot counts."""
    genus_match = defaultdict(int)
    genus_bdl_only = defaultdict(int)
    genus_pc_only = defaultdict(int)

    for pid in common_plots:
        bdl_genera = bdl_sets.get(pid, set())
        pc_genera = set(pc_counts.get(pid, {}).keys())

        for g in bdl_genera & pc_genera:
            genus_match[g] += 1
        for g in bdl_genera - pc_genera:
            genus_bdl_only[g] += 1
        for g in pc_genera - bdl_genera:
            genus_pc_only[g] += 1

    return genus_match, genus_bdl_only, genus_pc_only


def bdl_coverage_analysis(bdl_sets: dict, pc_counts: dict, common_plots: list):
    """For each genus, of the plots where PC has it, how many also have it in BDL?"""
    covered = defaultdict(int)
    total = defaultdict(int)

    for pid in common_plots:
        bdl_genera = bdl_sets.get(pid, set())
        pc_genera = set(pc_counts.get(pid, {}).keys())

        for g in pc_genera:
            total[g] += 1
            if g in bdl_genera:
                covered[g] += 1

    return covered, total


def pc_only_sample_counts(bdl_sets: dict, pc_counts: dict, common_plots: list):
    """For each PC-only occurrence (genus in PC but not BDL), collect sample counts."""
    genus_counts = defaultdict(list)

    for pid in common_plots:
        bdl_genera = bdl_sets.get(pid, set())
        pc = pc_counts.get(pid, {})

        for g, count in pc.items():
            if g not in bdl_genera:
                genus_counts[g].append(count)

    return genus_counts


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_presence_absence(pa, out_dir: Path):
    """Grouped bar chart: match/BDL-only/PC-only per genus."""
    match, bdl_only, pc_only = pa
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALL_GENERA))
    w = 0.25

    ax.bar(x - w, [match.get(g, 0) for g in ALL_GENERA], w,
           label="Both (match)", color="#2ca02c")
    ax.bar(x, [bdl_only.get(g, 0) for g in ALL_GENERA], w,
           label="BDL-only", color="#ff7f0e")
    ax.bar(x + w, [pc_only.get(g, 0) for g in ALL_GENERA], w,
           label="PC-only", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_GENERA, rotation=45, ha="right")
    ax.set_ylabel("Number of plots")
    ax.set_title("Presence / Absence: BDL vs Point Cloud")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "presence_absence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bdl_coverage(cov, out_dir: Path):
    """Per genus: of plots where PC has it, what % also have it in BDL?"""
    covered, total = cov
    fig, ax = plt.subplots(figsize=(10, 6))

    genera = [g for g in ALL_GENERA if total.get(g, 0) > 0]
    x = np.arange(len(genera))
    rates = [covered.get(g, 0) / total[g] for g in genera]
    totals = [total[g] for g in genera]

    ax.bar(x, rates, color=[GENUS_COLORS[g] for g in genera])
    for i, (r, t) in enumerate(zip(rates, totals)):
        ax.text(i, r + 0.02, f"{r:.0%}\n(n={t})", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genera, rotation=45, ha="right")
    ax.set_ylabel("BDL coverage rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("BDL Coverage of Point Cloud Species\n"
                 "(of plots where PC has genus X, how often does BDL also list it?)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / "bdl_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pc_only_counts(genus_counts, out_dir: Path):
    """Strip plot: when a genus is in PC but not BDL, how many trees per plot?"""
    genera = [g for g in ALL_GENERA if len(genus_counts.get(g, [])) > 0]
    if not genera:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(genera))

    for i, g in enumerate(genera):
        counts = genus_counts[g]
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(counts))
        ax.scatter(
            np.full(len(counts), i) + jitter, counts,
            alpha=0.6, s=30, color=GENUS_COLORS.get(g, "gray"),
            edgecolors="black", linewidths=0.3,
        )
        med = np.median(counts)
        ax.plot([i - 0.3, i + 0.3], [med, med], color="black", linewidth=1.5)
        ax.text(i, max(counts) + 1.5, f"n={len(counts)}", ha="center", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(genera, rotation=45, ha="right")
    ax.set_ylabel("Trees per plot (PC-only occurrences)")
    ax.set_title("PC-only Cases: How Many Trees When BDL Doesn't List the Genus?\n"
                 "(each dot = one plot; black line = median)")

    fig.tight_layout()
    fig.savefig(out_dir / "pc_only_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(out_dir: Path, n_bdl_plots, n_pc_plots, n_common, pa, cov, pconly):
    lines = []
    lines.append("# BDL vs Point Cloud Species Comparison\n")
    lines.append("**Question**: How well does BDL cover the species actually present in our point cloud plots?\n")
    lines.append("BDL describes entire forest subdivisions (potentially tens of hectares), "
                 "while our plots are ~500 m² circles. BDL listing extra species is expected. "
                 "The interesting cases are when **PC has a genus that BDL doesn't mention at all**.\n")

    lines.append("## Dataset\n")
    lines.append(f"- **BDL plots**: {n_bdl_plots}")
    lines.append(f"- **PC plots**: {n_pc_plots}")
    lines.append(f"- **Overlapping**: {n_common}")
    lines.append(f"- **Genera**: {len(ALL_GENERA)} ({', '.join(ALL_GENERA)})")
    lines.append("")

    # 1. Presence / Absence
    match, bdl_only, pc_only = pa
    lines.append("## 1. Presence / Absence Overview\n")
    lines.append("![Presence/Absence](presence_absence.png)\n")
    lines.append("| Genus | Both | BDL-only | PC-only |")
    lines.append("|-------|------|----------|---------|")
    for g in ALL_GENERA:
        lines.append(f"| {g} | {match.get(g, 0)} | {bdl_only.get(g, 0)} | {pc_only.get(g, 0)} |")
    lines.append("")

    # 2. BDL Coverage
    covered, total = cov
    lines.append("## 2. BDL Coverage of PC Species\n")
    lines.append("For each genus, of the plots where our point cloud contains it, "
                 "what fraction also has it listed in BDL?\n")
    lines.append("![BDL Coverage](bdl_coverage.png)\n")
    lines.append("| Genus | PC plots | BDL also lists | Coverage |")
    lines.append("|-------|----------|---------------|----------|")
    for g in ALL_GENERA:
        t = total.get(g, 0)
        c = covered.get(g, 0)
        rate = f"{c/t:.0%}" if t > 0 else "N/A"
        lines.append(f"| {g} | {t} | {c} | {rate} |")
    lines.append("")

    # 3. PC-only detail
    lines.append("## 3. PC-only Cases: How Many Trees?\n")
    lines.append("When the point cloud has a genus that BDL doesn't list, "
                 "is it a single stray tree or multiple?\n")
    lines.append("![PC-only counts](pc_only_counts.png)\n")
    lines.append("| Genus | Plots | Total trees | Median per plot | Max per plot |")
    lines.append("|-------|-------|-------------|----------------|-------------|")
    for g in ALL_GENERA:
        counts = pconly.get(g, [])
        if counts:
            lines.append(
                f"| {g} | {len(counts)} | {sum(counts)} "
                f"| {np.median(counts):.0f} | {max(counts)} |"
            )
    lines.append("")

    lines.append("---\n")
    lines.append("*BDL-only occurrences (BDL lists a genus not found in PC) are expected "
                 "because subdivisions are much larger than our ~500 m² plots.*\n")

    (out_dir / "bdl_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare BDL vs point cloud species data")
    parser.add_argument("--output_dir", type=str, default="results/bdl_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    bdl = load_bdl(BDL_PATH)
    pc_map = load_pc_mapping(PC_MAPPING_PATH)
    print(f"  BDL: {len(bdl)} rows, {bdl['num'].nunique()} plots")
    print(f"  PC:  {len(pc_map)} samples, {pc_map['plot_id'].nunique()} plots")

    # Unmapped BDL species
    unmapped = bdl[bdl["genus"] == "Other"]["sp_species_cd"].value_counts()
    if len(unmapped) > 0:
        print(f"\nUnmapped BDL species (excluded): {', '.join(f'{c}({n})' for c, n in unmapped.items())}")

    # Common plots
    bdl_plot_ids = set(bdl["num"].unique())
    pc_plot_ids = set(pc_map["plot_id"].unique())
    common_plots = sorted(bdl_plot_ids & pc_plot_ids)
    print(f"\nPlots: {len(common_plots)} common, "
          f"{len(bdl_plot_ids - pc_plot_ids)} BDL-only, "
          f"{len(pc_plot_ids - bdl_plot_ids)} PC-only")

    pc_cnts = pc_genus_counts(pc_map)
    bdl_sets = bdl_genus_sets(bdl)
    plots = [p for p in common_plots if p in bdl_sets]
    print(f"{len(plots)} plots with BDL genus data")

    pa = presence_absence_analysis(bdl_sets, pc_cnts, plots)
    cov = bdl_coverage_analysis(bdl_sets, pc_cnts, plots)
    pconly = pc_only_sample_counts(bdl_sets, pc_cnts, plots)

    total_pconly = sum(len(v) for v in pconly.values())
    print(f"PC-only occurrences: {total_pconly} (genus×plot pairs)")

    print("\nGenerating plots...")
    plot_presence_absence(pa, out_dir)
    plot_bdl_coverage(cov, out_dir)
    plot_pc_only_counts(pconly, out_dir)

    print("Generating report...")
    generate_report(out_dir, len(bdl_plot_ids), len(pc_plot_ids), len(common_plots),
                    pa, cov, pconly)

    print(f"Done! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
