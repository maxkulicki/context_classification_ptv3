"""
Habitat-based species presence baseline using only plot-level
fertility and moisture from BDL.

For each fertility x moisture combination, estimates the empirical
probability P(genus present | fertility, moisture) from training data.
At test time, all genera above a threshold are predicted as "present".
Each test sample's true genus is checked against the predicted set.

Evaluated under:
1. Plot-level train/test split (11 genera incl. Abies)
2. District-level 6-fold cross-validation (10 genera excl. Abies)

Outputs precision-recall curves, threshold analysis, and a markdown
report to results/naive_bayes/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

DATA_DIR = "Pointcept/data/treescanpl"
DISTRICTS = ["Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Suprasl"]


def load_data(fertility_moisture_csv, sample_plot_csv):
    """Load and merge fertility/moisture with sample-plot mapping."""
    fm = pd.read_csv(fertility_moisture_csv)
    sp = pd.read_csv(sample_plot_csv)
    df = sp.merge(fm[["plot_id", "district", "fertility", "moisture"]], on="plot_id", how="inner")
    df["genus"] = df["sample_name"].str.rsplit("_", n=1).str[0]
    return df


def load_split(split_file):
    """Load sample names from a split file."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def estimate_presence_probs(df_train, genera):
    """Estimate P(genus present | fertility, moisture) from training plots.

    For each (fertility, moisture) cell, compute the fraction of plots
    that contain at least one sample of each genus. Probabilities are
    independent per genus and can sum to more than 1.

    Returns a dict: (fertility, moisture) -> {genus: probability}
    """
    probs = {}
    for (fert, moist), group in df_train.groupby(["fertility", "moisture"]):
        plot_ids = group["plot_id"].unique()
        n_plots = len(plot_ids)
        cell_probs = {}
        for g in genera:
            plots_with_genus = group[group["genus"] == g]["plot_id"].nunique()
            cell_probs[g] = plots_with_genus / n_plots
        probs[(fert, moist)] = cell_probs
    return probs


def predict_at_threshold(probs, df_test, genera, threshold):
    """For each test sample, predict genera with P >= threshold.

    Returns per-sample: (true_genus, predicted_set, prob_of_true_genus)
    """
    results = []
    for _, row in df_test.iterrows():
        key = (row["fertility"], row["moisture"])
        if key not in probs:
            # Unseen combination — predict nothing
            pred_set = set()
            true_prob = 0.0
        else:
            cell_probs = probs[key]
            pred_set = {g for g, p in cell_probs.items() if p >= threshold}
            true_prob = cell_probs.get(row["genus"], 0.0)
        results.append(dict(
            sample_name=row["sample_name"],
            genus=row["genus"],
            predicted_set=pred_set,
            set_size=len(pred_set),
            hit=int(row["genus"] in pred_set),
            true_genus_prob=true_prob,
        ))
    return pd.DataFrame(results)


def sweep_thresholds(probs, df_test, genera, thresholds=None):
    """Sweep thresholds and compute metrics at each."""
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    records = []
    for t in thresholds:
        res = predict_at_threshold(probs, df_test, genera, t)
        n = len(res)
        coverage = res["hit"].mean()  # recall: fraction where true genus is in set
        mean_set_size = res["set_size"].mean()
        # Precision: across all (sample, genus) binary predictions
        # total positive predictions = sum of set sizes
        total_pos = res["set_size"].sum()
        true_pos = res["hit"].sum()
        precision = true_pos / total_pos if total_pos > 0 else 0.0
        f1 = 2 * precision * coverage / (precision + coverage) if (precision + coverage) > 0 else 0.0

        # Per-genus metrics (binary: present/not-present for each genus)
        genus_stats = {}
        for g in genera:
            # True positives: samples of genus g where g is in predicted set
            tp = res[res["genus"] == g]["hit"].sum()
            # False negatives: samples of genus g where g is NOT in predicted set
            fn = len(res[res["genus"] == g]) - tp
            # False positives: samples of OTHER genera where g is in predicted set
            fp = res[res["genus"] != g].apply(
                lambda r: int(g in r["predicted_set"]), axis=1
            ).sum() if total_pos > 0 else 0
            g_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            g_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
            genus_stats[g] = dict(precision=g_prec, recall=g_rec, f1=g_f1,
                                  support=int(tp + fn))

        macro_f1 = np.mean([s["f1"] for s in genus_stats.values()])

        records.append(dict(
            threshold=t,
            coverage=coverage,
            precision=precision,
            f1=f1,
            macro_f1=macro_f1,
            mean_set_size=mean_set_size,
            genus_stats=genus_stats,
        ))

    return pd.DataFrame(records)


def evaluate_split(df, split_name, train_names, test_names, genera, output_dir):
    """Estimate presence probs from train, sweep thresholds on test."""
    df_train = df[df["sample_name"].isin(set(train_names)) & df["genus"].isin(genera)]
    df_test = df[df["sample_name"].isin(set(test_names)) & df["genus"].isin(genera)]

    print(f"\n{'='*60}")
    print(f"{split_name}: {len(df_train)} train, {len(df_test)} test samples")
    print(f"{'='*60}")

    if len(df_test) == 0:
        print("  No test samples, skipping.")
        return None

    probs = estimate_presence_probs(df_train, genera)

    # Print the probability table
    print("\n  P(genus | fertility, moisture):")
    prob_rows = []
    for (fert, moist), cell in sorted(probs.items()):
        for g, p in sorted(cell.items()):
            prob_rows.append(dict(fertility=fert, moisture=moist, genus=g, prob=p))
    prob_df = pd.DataFrame(prob_rows)
    pivot = prob_df.pivot_table(values="prob", index="genus",
                                 columns=["fertility", "moisture"])
    print(pivot.round(3).to_string(col_space=8))

    # Sweep thresholds
    metrics = sweep_thresholds(probs, df_test, genera)

    # Best macro F1
    best_idx = metrics["macro_f1"].idxmax()
    best = metrics.loc[best_idx]
    print(f"\n  Best macro F1: {best['macro_f1']:.4f} at threshold={best['threshold']:.2f}")
    print(f"  Micro F1: {best['f1']:.4f}")
    print(f"  Coverage (recall): {best['coverage']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Mean set size: {best['mean_set_size']:.2f}")

    return dict(
        split_name=split_name,
        n_train=len(df_train),
        n_test=len(df_test),
        probs=probs,
        metrics=metrics,
        best_threshold=best["threshold"],
        best_macro_f1=best["macro_f1"],
        best_f1=best["f1"],
        best_coverage=best["coverage"],
        best_precision=best["precision"],
        best_set_size=best["mean_set_size"],
        genera=genera,
    )


def plot_pr_curve(results_list, output_dir):
    """Plot precision-recall curves for all splits."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(7 * len(results_list), 5),
                              squeeze=False)
    for i, res in enumerate(results_list):
        ax = axes[0, i]
        m = res["metrics"]
        ax.plot(m["coverage"], m["precision"], "b-", linewidth=2)
        # Mark best macro F1
        best_idx = m["macro_f1"].idxmax()
        ax.plot(m.loc[best_idx, "coverage"], m.loc[best_idx, "precision"],
                "r*", markersize=15, label=f"Best macro F1={m.loc[best_idx, 'macro_f1']:.3f}\n"
                f"(t={m.loc[best_idx, 'threshold']:.2f})")
        ax.set_xlabel("Coverage (Recall)")
        ax.set_ylabel("Precision")
        ax.set_title(res["split_name"])
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=150)
    plt.close(fig)


def plot_threshold_analysis(results_list, output_dir):
    """Plot F1, coverage, precision, set size vs threshold."""
    fig, axes = plt.subplots(len(results_list), 1,
                              figsize=(10, 5 * len(results_list)), squeeze=False)
    for i, res in enumerate(results_list):
        ax = axes[i, 0]
        m = res["metrics"]
        ax.plot(m["threshold"], m["macro_f1"], "r-", linewidth=2, label="Macro F1")
        ax.plot(m["threshold"], m["f1"], "r:", linewidth=1, alpha=0.5, label="Micro F1")
        ax.plot(m["threshold"], m["coverage"], "b--", linewidth=1.5, label="Coverage (recall)")
        ax.plot(m["threshold"], m["precision"], "g--", linewidth=1.5, label="Precision")

        ax2 = ax.twinx()
        ax2.plot(m["threshold"], m["mean_set_size"], "k:", linewidth=1, alpha=0.5,
                 label="Mean set size")
        ax2.set_ylabel("Mean set size", color="gray")

        # Mark best macro F1
        best_idx = m["macro_f1"].idxmax()
        ax.axvline(m.loc[best_idx, "threshold"], color="red", alpha=0.3, linestyle="--")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(res["split_name"])
        ax.legend(loc="center left")
        ax2.legend(loc="center right")
        ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_analysis.png"), dpi=150)
    plt.close(fig)


def plot_genus_f1(results_list, output_dir):
    """Per-genus precision, recall, F1 at best macro F1 threshold."""
    for res in results_list:
        m = res["metrics"]
        best_idx = m["macro_f1"].idxmax()
        gs = m.loc[best_idx, "genus_stats"]
        genera = res["genera"]

        # Sort by F1
        sorted_genera = sorted(genera, key=lambda g: gs[g]["f1"], reverse=True)

        x = np.arange(len(sorted_genera))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        prec_vals = [gs[g]["precision"] for g in sorted_genera]
        rec_vals = [gs[g]["recall"] for g in sorted_genera]
        f1_vals = [gs[g]["f1"] for g in sorted_genera]

        ax.bar(x - width, prec_vals, width, label="Precision", color="#3498db")
        ax.bar(x, rec_vals, width, label="Recall", color="#2ecc71")
        ax.bar(x + width, f1_vals, width, label="F1", color="#e74c3c")

        for i, g in enumerate(sorted_genera):
            ax.text(i + width, f1_vals[i] + 0.02, f"{f1_vals[i]:.2f}",
                    ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_genera, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"{res['split_name']} — Per-genus Metrics "
                     f"(threshold={res['best_threshold']:.2f})")
        ax.set_ylim(0, 1.15)
        ax.legend()
        plt.tight_layout()
        safe = res["split_name"].replace(" ", "_").lower()
        fig.savefig(os.path.join(output_dir, f"genus_f1_{safe}.png"), dpi=150)
        plt.close(fig)


def plot_probability_heatmap(probs, genera, title, path):
    """Heatmap of P(genus | fertility, moisture) for each cell."""
    rows = []
    for (fert, moist), cell in sorted(probs.items()):
        for g in genera:
            rows.append(dict(condition=f"{fert}\n{moist}", genus=g, prob=cell.get(g, 0)))
    hdf = pd.DataFrame(rows)
    pivot = hdf.pivot_table(values="prob", index="genus", columns="condition")
    # Sort by mean probability
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=pivot.max().max() + 0.05,
                linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Genus")
    ax.set_xlabel("Site Condition")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_report(all_results, output_dir):
    """Generate markdown report."""
    lines = [
        "# Habitat-based Species Presence Baseline",
        "",
        "## Method",
        "",
        "For each fertility x moisture combination, estimate the empirical probability",
        "P(genus present | fertility, moisture) from training sample co-occurrence frequencies.",
        "At test time, every genus with probability >= threshold is predicted as present.",
        "A test sample is a **hit** if its true genus is in the predicted set.",
        "",
        "Features:",
        "- **Fertility**: oligotrophic / mesotrophic / mesoeutrophic / eutrophic",
        "- **Moisture**: fresh / moist_or_wet",
        "- **8 possible site condition cells** (4 x 2)",
        "",
        "No point cloud information is used.",
        "",
    ]

    for res in all_results:
        safe = res["split_name"].replace(" ", "_").lower()
        m = res["metrics"]
        best_idx = m["macro_f1"].idxmax()
        best = m.loc[best_idx]
        gs = best["genus_stats"]

        lines.extend([
            f"## {res['split_name']}",
            "",
            f"- **Train**: {res['n_train']} samples",
            f"- **Test**: {res['n_test']} samples",
            "",
            f"### Best Macro F1 Operating Point",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threshold | {best['threshold']:.2f} |",
            f"| Macro F1 | {best['macro_f1']:.4f} |",
            f"| Micro F1 | {best['f1']:.4f} |",
            f"| Coverage (recall) | {best['coverage']:.4f} |",
            f"| Precision | {best['precision']:.4f} |",
            f"| Mean set size | {best['mean_set_size']:.2f} |",
            "",
            "### Per-genus Performance at Best Threshold",
            "",
            "| Genus | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|----|---------|",
        ])
        for g in sorted(gs.keys()):
            s = gs[g]
            lines.append(f"| {g} | {s['precision']:.4f} | {s['recall']:.4f} | "
                         f"{s['f1']:.4f} | {s['support']} |")

        lines.extend([
            "",
            f"![Probability Heatmap](prob_heatmap_{safe}.png)",
            "",
            f"![Per-genus F1](genus_f1_{safe}.png)",
            "",
        ])

    lines.extend([
        "## Threshold Analysis",
        "",
        "![Threshold Analysis](threshold_analysis.png)",
        "",
        "## Precision-Recall Curves",
        "",
        "![PR Curves](precision_recall_curves.png)",
        "",
        "## Key Takeaways",
        "",
        "- With only 8 possible site condition cells for 10-11 genera, this baseline",
        "  can only distinguish genera with strong habitat preferences.",
        "- Coverage (recall) is high at low thresholds but at the cost of large predicted",
        "  sets, indicating low discriminative power.",
        "- The F1-optimal threshold reveals the fundamental limit of what fertility and",
        "  moisture alone can predict about species composition.",
    ])

    report_path = os.path.join(output_dir, "naive_bayes_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fertility_moisture",
                        default="/home/makskulicki/tree_species_context_classification/data/plots_fertility_moisture.csv")
    parser.add_argument("--sample_plot",
                        default=os.path.join(DATA_DIR, "sample_plotid_mapping.csv"))
    parser.add_argument("--output_dir", default="results/naive_bayes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.fertility_moisture, args.sample_plot)
    print(f"Loaded {len(df)} samples across {df['plot_id'].nunique()} plots")
    print(f"Genera: {sorted(df['genus'].unique())}")
    print(f"Fertility: {sorted(df['fertility'].unique())}")
    print(f"Moisture: {sorted(df['moisture'].unique())}")

    all_results = []

    # ── 1. Plot-level split (11 genera incl. Abies) ──
    genera_11 = sorted(df["genus"].unique())
    train_names = load_split(os.path.join(DATA_DIR, "treescanpl_train.txt"))
    test_names = load_split(os.path.join(DATA_DIR, "treescanpl_test.txt"))
    res = evaluate_split(df, "Plot-level split", train_names, test_names,
                         genera_11, args.output_dir)
    if res:
        all_results.append(res)
        plot_probability_heatmap(
            res["probs"], genera_11, "P(genus | fertility, moisture) — Plot-level split",
            os.path.join(args.output_dir, "prob_heatmap_plot-level_split.png"),
        )

    # ── 2. District k-fold (10 genera excl. Abies) ──
    genera_10 = [g for g in genera_11 if g != "Abies"]

    # Aggregate k-fold: collect all fold results, then merge metrics
    kfold_results_per_fold = []
    kfold_all_hits = []
    kfold_all_set_sizes = []

    for fold in range(6):
        district = DISTRICTS[fold]
        train_names = load_split(os.path.join(DATA_DIR, f"treescanpl_fold{fold}_train.txt"))
        test_names = load_split(os.path.join(DATA_DIR, f"treescanpl_fold{fold}_test.txt"))
        res = evaluate_split(df, f"Fold {fold} ({district})", train_names, test_names,
                             genera_10, args.output_dir)
        if res:
            kfold_results_per_fold.append(res)

    # Aggregate k-fold: train per-fold, predict per-fold, combine test predictions
    # then sweep thresholds on the combined predictions
    print(f"\n{'='*60}")
    print("K-fold aggregate (6 districts)")
    print(f"{'='*60}")

    # Pre-load fold test DataFrames
    fold_test_dfs = {}
    for fold in range(6):
        test_names = load_split(os.path.join(DATA_DIR, f"treescanpl_fold{fold}_test.txt"))
        fold_test_dfs[fold] = df[df["sample_name"].isin(set(test_names)) & df["genus"].isin(genera_10)]

    # Pre-compute per-sample probabilities for each genus across all folds
    # so we can threshold efficiently without re-running predict_at_threshold
    sample_records = []
    for fold, fold_res in enumerate(kfold_results_per_fold):
        probs = fold_res["probs"]
        for _, row in fold_test_dfs[fold].iterrows():
            key = (row["fertility"], row["moisture"])
            cell = probs.get(key, {})
            rec = {"genus": row["genus"]}
            for g in genera_10:
                rec[f"prob_{g}"] = cell.get(g, 0.0)
            sample_records.append(rec)
    samples_df = pd.DataFrame(sample_records)
    true_genera = samples_df["genus"].values
    prob_matrix = samples_df[[f"prob_{g}" for g in genera_10]].values  # [N, G]

    thresholds = np.arange(0.0, 1.01, 0.01)
    agg_records = []
    for t in thresholds:
        predicted = prob_matrix >= t  # [N, G] bool
        n_samples = len(true_genera)
        set_sizes = predicted.sum(axis=1)
        mean_ss = set_sizes.mean()
        total_pos = set_sizes.sum()

        # Per-genus stats
        genus_stats = {}
        total_tp = 0
        total_hits = 0
        for gi, g in enumerate(genera_10):
            is_true = (true_genera == g)
            is_pred = predicted[:, gi]
            tp = int((is_true & is_pred).sum())
            fp = int((~is_true & is_pred).sum())
            fn = int((is_true & ~is_pred).sum())
            g_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            g_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
            genus_stats[g] = dict(precision=g_prec, recall=g_rec, f1=g_f1,
                                  support=int(tp + fn))
            total_tp += tp

        coverage = total_tp / n_samples if n_samples > 0 else 0.0
        precision = total_tp / total_pos if total_pos > 0 else 0.0
        f1 = 2 * precision * coverage / (precision + coverage) if (precision + coverage) > 0 else 0.0

        macro_f1 = np.mean([s["f1"] for s in genus_stats.values()])

        agg_records.append(dict(
            threshold=t, coverage=coverage, precision=precision,
            f1=f1, macro_f1=macro_f1, mean_set_size=mean_ss, genus_stats=genus_stats,
        ))

    agg_metrics = pd.DataFrame(agg_records)
    best_idx = agg_metrics["macro_f1"].idxmax()
    best = agg_metrics.loc[best_idx]
    print(f"  Best macro F1: {best['macro_f1']:.4f} at threshold={best['threshold']:.2f}")
    print(f"  Micro F1: {best['f1']:.4f}")
    print(f"  Coverage (recall): {best['coverage']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Mean set size: {best['mean_set_size']:.2f}")

    # Use fold 0's probs for the heatmap (representative)
    if kfold_results_per_fold:
        plot_probability_heatmap(
            kfold_results_per_fold[0]["probs"], genera_10,
            "P(genus | fertility, moisture) — K-fold (Fold 0 example)",
            os.path.join(args.output_dir, "prob_heatmap_kfold_aggregate.png"),
        )

    kfold_agg = dict(
        split_name="K-fold aggregate",
        n_train="~5300 per fold",
        n_test=sum(r["n_test"] for r in kfold_results_per_fold),
        probs=kfold_results_per_fold[0]["probs"] if kfold_results_per_fold else {},
        metrics=agg_metrics,
        best_threshold=best["threshold"],
        best_macro_f1=best["macro_f1"],
        best_f1=best["f1"],
        best_coverage=best["coverage"],
        best_precision=best["precision"],
        best_set_size=best["mean_set_size"],
        genera=genera_10,
    )
    all_results.append(kfold_agg)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_pr_curve(all_results, args.output_dir)
    plot_threshold_analysis(all_results, args.output_dir)
    plot_genus_f1(all_results, args.output_dir)

    # ── Report ──
    generate_report(all_results, args.output_dir)
    print(f"All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
