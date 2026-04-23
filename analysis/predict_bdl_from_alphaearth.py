"""
Can AlphaEarth embeddings predict BDL fertility and moisture?

If yes, BDL site data is redundant — the fusion model already has
this information via AlphaEarth features.

Trains Random Forest and Logistic Regression on plot-level data
to predict fertility (4 classes) and moisture (2 classes) from
64-dim AlphaEarth embeddings. Evaluated with stratified k-fold CV
and district-level leave-one-out CV.

Outputs results + plots to results/ae_predicts_bdl/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.0)

AE_FEATURES = [f"A{i:02d}" for i in range(64)]


def load_data(ae_csv, fm_csv):
    """Load and merge AlphaEarth features with fertility/moisture labels."""
    ae = pd.read_csv(ae_csv, sep=";")
    ae = ae.rename(columns={"num": "plot_id"})
    fm = pd.read_csv(fm_csv)

    df = ae.merge(fm[["plot_id", "district", "fertility", "moisture"]], on="plot_id", how="inner")
    print(f"Merged: {len(df)} plots with both AE features and BDL labels")
    return df


def evaluate_cv(X, y, target_name, class_names, cv_splits, output_dir):
    """Evaluate RF and LogReg with given CV splits."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
    }

    results = {}
    for model_name, clf_template in models.items():
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in cv_splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = type(clf_template)(**clf_template.get_params())
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        acc = accuracy_score(all_y_true, all_y_pred)
        bacc = balanced_accuracy_score(all_y_true, all_y_pred)

        results[model_name] = dict(
            accuracy=acc,
            balanced_accuracy=bacc,
            y_true=all_y_true,
            y_pred=all_y_pred,
        )

        print(f"\n  {model_name}:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Balanced accuracy: {bacc:.4f}")
        print(classification_report(all_y_true, all_y_pred, zero_division=0))

    return results


def plot_confusion_matrices(results, class_names, target_name, cv_name, output_dir):
    """Plot confusion matrices for all models side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidths=0.5, ax=ax)
        # Add raw counts as secondary annotation
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.75, f"({cm[i, j]})",
                        ha="center", va="center", fontsize=8, color="gray")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name}\nacc={res['accuracy']:.3f}, "
                     f"bal_acc={res['balanced_accuracy']:.3f}")

    fig.suptitle(f"Predicting {target_name} from AlphaEarth ({cv_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = f"{target_name}_{cv_name}".replace(" ", "_").lower()
    fig.savefig(os.path.join(output_dir, f"cm_{safe}.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_csv",
                        default="Pointcept/data/treescanpl/plots_alphaearth_2018.csv")
    parser.add_argument("--fm_csv",
                        default="/home/makskulicki/tree_species_context_classification/data/plots_fertility_moisture.csv")
    parser.add_argument("--output_dir", default="results/ae_predicts_bdl")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.ae_csv, args.fm_csv)
    X = df[AE_FEATURES].values

    print(f"\nFertility distribution:\n{df['fertility'].value_counts().to_string()}")
    print(f"\nMoisture distribution:\n{df['moisture'].value_counts().to_string()}")

    targets = {
        "fertility": {
            "classes": ["oligotrophic", "mesotrophic", "mesoeutrophic", "eutrophic"],
        },
        "moisture": {
            "classes": ["fresh", "moist_or_wet"],
        },
    }

    all_results = {}

    for target_name, cfg in targets.items():
        y = df[target_name].values
        class_names = cfg["classes"]

        # ── 1. Stratified 5-fold CV ──
        print(f"\n{'='*60}")
        print(f"Target: {target_name} — Stratified 5-fold CV")
        print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(skf.split(X, y))

        results_skf = evaluate_cv(X, y, target_name, class_names, cv_splits, args.output_dir)
        plot_confusion_matrices(results_skf, class_names, target_name, "Stratified 5-fold", args.output_dir)

        # ── 2. District leave-one-out CV ──
        print(f"\n{'='*60}")
        print(f"Target: {target_name} — District leave-one-out CV")
        print(f"{'='*60}")

        districts = df["district"].values
        unique_districts = sorted(df["district"].unique())
        district_splits = []
        for d in unique_districts:
            test_mask = districts == d
            train_mask = ~test_mask
            district_splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

        results_dist = evaluate_cv(X, y, target_name, class_names, district_splits, args.output_dir)
        plot_confusion_matrices(results_dist, class_names, target_name, "District LOO", args.output_dir)

        all_results[target_name] = dict(stratified=results_skf, district=results_dist)

    # ── Feature importance (RF on full data) ──
    print(f"\n{'='*60}")
    print("Feature importance (RF trained on all data)")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, target_name in zip(axes, targets.keys()):
        y = df[target_name].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_s, y)

        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-15:]
        ax.barh([AE_FEATURES[i] for i in top_idx], importances[top_idx], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 features for {target_name}")

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # ── Generate report ──
    lines = [
        "# Can AlphaEarth Predict BDL Fertility & Moisture?",
        "",
        "## Question",
        "",
        "If AlphaEarth satellite embeddings (64-dim) already encode plot-level",
        "fertility and moisture, then BDL site data adds no new information to",
        "the fusion model — it's already captured in the AE stream.",
        "",
        f"**Dataset**: {len(df)} plots with both AE features and BDL labels.",
        "",
    ]

    for target_name, cfg in targets.items():
        class_names = cfg["classes"]
        lines.extend([
            f"## {target_name.capitalize()}",
            "",
            f"Classes: {', '.join(class_names)} "
            f"({', '.join(str(c) for c in df[target_name].value_counts().sort_index().values)})",
            "",
        ])

        for cv_name, cv_key in [("Stratified 5-fold", "stratified"), ("District LOO", "district")]:
            results = all_results[target_name][cv_key]
            safe = f"{target_name}_{cv_name}".replace(" ", "_").lower()
            lines.extend([
                f"### {cv_name}",
                "",
                "| Model | Accuracy | Balanced Accuracy |",
                "|-------|----------|-------------------|",
            ])
            for model_name, res in results.items():
                lines.append(f"| {model_name} | {res['accuracy']:.4f} | "
                             f"{res['balanced_accuracy']:.4f} |")
            lines.extend([
                "",
                f"![Confusion Matrices](cm_{safe}.png)",
                "",
            ])

    # Chance baselines
    fert_chance = 1.0 / len(targets["fertility"]["classes"])
    moist_majority = df["moisture"].value_counts().max() / len(df)

    lines.extend([
        "## Feature Importance (Random Forest)",
        "",
        "![Feature Importance](feature_importance.png)",
        "",
        "## Summary",
        "",
        f"- **Fertility chance baseline**: {fert_chance:.1%} (uniform), "
        f"majority class: {df['fertility'].value_counts().idxmax()} "
        f"({df['fertility'].value_counts().max()/len(df):.1%})",
        f"- **Moisture chance baseline**: majority class {df['moisture'].value_counts().idxmax()} "
        f"({moist_majority:.1%})",
        "",
    ])

    # Conclusion
    fert_best = max(
        all_results["fertility"]["district"]["Random Forest"]["balanced_accuracy"],
        all_results["fertility"]["district"]["Logistic Regression"]["balanced_accuracy"],
    )
    moist_best = max(
        all_results["moisture"]["district"]["Random Forest"]["balanced_accuracy"],
        all_results["moisture"]["district"]["Logistic Regression"]["balanced_accuracy"],
    )

    lines.extend([
        "## Conclusion",
        "",
        f"District-LOO balanced accuracy: **fertility={fert_best:.1%}**, "
        f"**moisture={moist_best:.1%}**.",
        "",
    ])

    if fert_best > 0.6 and moist_best > 0.7:
        lines.append(
            "AlphaEarth embeddings can reliably predict both fertility and moisture, "
            "suggesting BDL site data is largely redundant — the information is already "
            "encoded in the satellite features."
        )
    elif fert_best > 0.4 or moist_best > 0.6:
        lines.append(
            "AlphaEarth embeddings capture some but not all of the fertility/moisture signal. "
            "BDL may still provide complementary information for certain site conditions."
        )
    else:
        lines.append(
            "AlphaEarth embeddings cannot reliably predict fertility or moisture, "
            "suggesting BDL provides genuinely new information about site conditions."
        )

    report_path = os.path.join(args.output_dir, "ae_predicts_bdl_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport: {report_path}")
    print(f"All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
