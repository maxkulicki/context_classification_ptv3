"""
Analyze how point cloud size (height, crown area) affects classification accuracy
across models, genera, and datasets.

Prerequisites:
  1. Run compute_tree_geometry.py to produce data/tree_geometry.csv
  2. Have predictions.csv files in Pointcept/eval_results/{exp_name}/

Usage:
    conda run -n context_baseline python analyze_size_vs_accuracy.py \
        [--models model_a model_b ...] \
        [--outdir Pointcept/eval_results/size_analysis]

Default models (all that have predictions.csv with 2331 rows):
    ptv3_baseline_120ep
    ptv3_sinr_gauss_120ep
    ptv3_ae_combined_aug_vmf_120ep
    ptv3_ae_sinr_cat_aux_200ep
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path('/net/pr2/projects/plgrid/plggtreeseg')
EVAL_DIR = ROOT / 'context_classification_ptv3/Pointcept/eval_results'
GEOMETRY_CSV = ROOT / 'context_classification_ptv3/data/tree_geometry.csv'

DEFAULT_MODELS = [
    'ptv3_baseline_120ep',
    'ptv3_sinr_gauss_120ep',
    'ptv3_ae_combined_aug_vmf_120ep',
    'ptv3_ae_sinr_cat_aux_200ep',
]

MODEL_LABELS = {
    'ptv3_baseline_120ep': 'Baseline',
    'ptv3_sinr_gauss_120ep': 'SINR',
    'ptv3_ae_combined_aug_vmf_120ep': 'AE',
    'ptv3_ae_sinr_cat_aux_200ep': 'AE+SINR',
}

N_BINS = 4
BIN_LABELS = ['Q1\n(smallest)', 'Q2', 'Q3', 'Q4\n(largest)']
BIN_LABELS_SHORT = ['Q1', 'Q2', 'Q3', 'Q4']

GENERA = ['Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
          'Fagus', 'Larix', 'Picea', 'Pinus', 'Quercus']

MODEL_COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
PALETTE = dict(zip(DEFAULT_MODELS, MODEL_COLORS))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_geometry():
    df = pd.read_csv(GEOMETRY_CSV)
    # Only test split has predictions
    return df


def load_predictions(models, split_subdir=''):
    """Load predictions.csv for each model.
    split_subdir: subdirectory under eval_results (e.g. 'val_ood').
    Empty string means root eval_results (test split).
    """
    frames = []
    base = EVAL_DIR / split_subdir if split_subdir else EVAL_DIR
    for model in models:
        p = base / model / 'predictions.csv'
        if not p.exists():
            print(f'  WARNING: no predictions.csv for {model} (looked in {p})')
            continue
        df = pd.read_csv(p)
        df['model'] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No predictions.csv found under {base}')
    return pd.concat(frames, ignore_index=True)


def merge(geom_df, pred_df, geom_split='test'):
    split_geom = geom_df[geom_df['split'] == geom_split].copy()
    merged = pred_df.merge(
        split_geom[['stem', 'height', 'crown_area', 'hull_degenerate']],
        on='stem', how='left'
    )
    return merged


def add_bins(df, col, n=N_BINS, suffix='_bin'):
    """Add quantile bin column. Bins computed on all non-NaN rows."""
    valid = df[col].dropna()
    quantiles = np.linspace(0, 100, n + 1)
    edges = np.unique(np.percentile(valid, quantiles))
    if len(edges) < 2:
        df[col + suffix] = 0
        return df, ['all']
    labels = [f'Q{i+1}' for i in range(len(edges) - 1)]
    df[col + suffix] = pd.cut(df[col], bins=edges, labels=labels,
                               include_lowest=True)
    return df, labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def model_label(m):
    return MODEL_LABELS.get(m, m)


def model_color(m, models):
    idx = models.index(m) if m in models else 0
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Plot 1 & 2: Accuracy vs. size bins per genus
# ---------------------------------------------------------------------------

def plot_accuracy_vs_bins(df, bin_col, bin_labels, models, title_prefix, out_path):
    genera_present = [g for g in GENERA if g in df['true_genus'].unique()]
    ncols = 5
    nrows = int(np.ceil(len(genera_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3),
                              sharey=True)
    axes = axes.flatten()

    x = np.arange(len(bin_labels))
    width = 0.8 / len(models)

    for i, genus in enumerate(genera_present):
        ax = axes[i]
        gdf = df[df['true_genus'] == genus]
        for j, model in enumerate(models):
            mdf = gdf[gdf['model'] == model]
            accs = []
            ns = []
            for bl in bin_labels:
                bdf = mdf[mdf[bin_col] == bl]
                n = len(bdf)
                acc = bdf['correct'].mean() if n > 0 else np.nan
                accs.append(acc)
                ns.append(n)
            offset = (j - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, accs, width * 0.9,
                          color=model_color(model, models),
                          label=model_label(model), alpha=0.85)
        ax.set_title(genus, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS_SHORT, fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='grey', lw=0.5, ls='--')
        if i % ncols == 0:
            ax.set_ylabel('Accuracy')

    # Hide unused axes
    for i in range(len(genera_present), len(axes)):
        axes[i].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=model_color(m, models), label=model_label(m))
               for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=len(models),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(f'{title_prefix} — accuracy by {bin_col.replace("_bin","")} quartile (Q1=smallest)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3: Scatter height vs crown area, coloured by correct/incorrect
# ---------------------------------------------------------------------------

def plot_scatter_size_errors(df, models, out_path):
    nrows, ncols = 1, len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, 4), sharey=True, sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = df[df['model'] == model].dropna(subset=['height', 'crown_area'])
        correct = mdf[mdf['correct'] == 1]
        wrong = mdf[mdf['correct'] == 0]
        ax.scatter(correct['crown_area'], correct['height'],
                   s=4, alpha=0.25, color='#4878CF', label='Correct', rasterized=True)
        ax.scatter(wrong['crown_area'], wrong['height'],
                   s=6, alpha=0.5, color='#D65F5F', label='Wrong', rasterized=True)
        ax.set_title(model_label(model), fontsize=10)
        ax.set_xlabel('Crown area (m²)')
        if ax == axes[0]:
            ax.set_ylabel('Height (m)')

    handles = [mpatches.Patch(color='#4878CF', label='Correct'),
               mpatches.Patch(color='#D65F5F', label='Wrong')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle('Errors in tree size space', fontsize=12)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 4: Size distributions per dataset
# ---------------------------------------------------------------------------

def plot_size_distributions(geom_df, out_path):
    test = geom_df[geom_df['split'] == 'test'].copy()
    datasets = sorted(test['dataset'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label in zip(axes, ['height', 'crown_area'],
                               ['Height (m)', 'Crown area (m²)']):
        data = [test[test['dataset'] == ds][col].dropna().values for ds in datasets]
        bp = ax.boxplot(data, labels=datasets, patch_artist=True, notch=False)
        for patch, color in zip(bp['boxes'],
                                 plt.cm.Set2(np.linspace(0, 1, len(datasets)))):
            patch.set_facecolor(color)
        ax.set_ylabel(label)
        ax.set_xlabel('Dataset')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Tree size distributions by dataset (test set)', fontsize=12)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 5: Delta accuracy (model vs. baseline) by size bin
# ---------------------------------------------------------------------------

def plot_delta_accuracy(df, bin_col, bin_labels, models, baseline_model, out_path):
    genera_present = [g for g in GENERA if g in df['true_genus'].unique()]
    # Collapse over genera + one panel per genus
    panels = ['ALL'] + genera_present
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                              sharey=False)
    axes = axes.flatten()
    other_models = [m for m in models if m != baseline_model]

    x = np.arange(len(bin_labels))
    width = 0.8 / len(other_models)

    for i, panel in enumerate(panels):
        ax = axes[i]
        if panel == 'ALL':
            gdf = df
        else:
            gdf = df[df['true_genus'] == panel]

        bdf_base = gdf[gdf['model'] == baseline_model]
        base_accs = {}
        for bl in bin_labels:
            bdf = bdf_base[bdf_base[bin_col] == bl]
            base_accs[bl] = bdf['correct'].mean() if len(bdf) > 0 else np.nan

        for j, model in enumerate(other_models):
            mdf = gdf[gdf['model'] == model]
            deltas = []
            for bl in bin_labels:
                bdf = mdf[mdf[bin_col] == bl]
                acc = bdf['correct'].mean() if len(bdf) > 0 else np.nan
                deltas.append(acc - base_accs[bl] if not np.isnan(acc) else np.nan)
            offset = (j - len(other_models) / 2 + 0.5) * width
            ax.bar(x + offset, deltas, width * 0.9,
                   color=model_color(model, models),
                   label=model_label(model), alpha=0.85)

        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(panel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS_SHORT, fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel('∆ accuracy vs. baseline')

    for i in range(len(panels), len(axes)):
        axes[i].set_visible(False)

    handles = [mpatches.Patch(color=model_color(m, models), label=model_label(m))
               for m in other_models]
    fig.legend(handles=handles, loc='lower center', ncol=len(other_models),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        f'∆ accuracy vs. {model_label(baseline_model)} by {bin_col.replace("_bin","")} quartile\n'
        f'Positive = context model beats baseline for that size bin',
        fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 6: Size distributions per genus
# ---------------------------------------------------------------------------

def plot_size_by_genus(geom_df, out_path):
    test = geom_df[geom_df['split'] == 'test'].copy()
    genera_present = [g for g in GENERA if g in test['genus'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label in zip(axes, ['height', 'crown_area'],
                               ['Height (m)', 'Crown area (m²)']):
        data = [test[test['genus'] == g][col].dropna().values for g in genera_present]
        ns = [len(d) for d in data]
        xlabels = [f'{g}\n(n={n})' for g, n in zip(genera_present, ns)]
        bp = ax.boxplot(data, labels=xlabels, patch_artist=True, notch=False)
        colors = plt.cm.tab10(np.linspace(0, 1, len(genera_present)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Tree size distributions by genus (test set)', fontsize=12)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 7: Per-dataset accuracy vs. size bins (across models)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_bins_by_dataset(df, bin_col, bin_labels, models, title_prefix, out_path):
    datasets = sorted(df['dataset'].unique())
    ncols = min(len(datasets), 4)
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                              sharey=True)
    axes = np.array(axes).flatten()

    x = np.arange(len(bin_labels))
    width = 0.8 / len(models)

    for i, ds in enumerate(datasets):
        ax = axes[i]
        ddf = df[df['dataset'] == ds]
        for j, model in enumerate(models):
            mdf = ddf[ddf['model'] == model]
            accs = []
            for bl in bin_labels:
                bdf = mdf[mdf[bin_col] == bl]
                acc = bdf['correct'].mean() if len(bdf) > 0 else np.nan
                accs.append(acc)
            offset = (j - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, accs, width * 0.9,
                   color=model_color(model, models),
                   label=model_label(model), alpha=0.85)
        n_total = len(ddf) // len(models)
        ax.set_title(f'{ds} (n={n_total})', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS_SHORT, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='grey', lw=0.5, ls='--')
        if i % ncols == 0:
            ax.set_ylabel('Accuracy')

    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)

    handles = [mpatches.Patch(color=model_color(m, models), label=model_label(m))
               for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=len(models),
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(f'{title_prefix} — accuracy by {bin_col.replace("_bin","")} quartile per dataset',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary(df, models, out_path):
    rows = []
    for metric_col in ['height_bin', 'crown_area_bin']:
        for group_col, groups in [('true_genus', GENERA), ('dataset', sorted(df['dataset'].unique()))]:
            for group_val in groups:
                gdf = df[df[group_col] == group_val] if group_val != 'ALL' else df
                if len(gdf) == 0:
                    continue
                for bin_val in gdf[metric_col].dropna().unique():
                    bdf = gdf[gdf[metric_col] == bin_val]
                    row = {'metric': metric_col.replace('_bin',''),
                           'group_by': group_col, 'group': group_val,
                           'bin': str(bin_val), 'n': len(bdf) // len(models)}
                    for model in models:
                        mdf = bdf[bdf['model'] == model]
                        row[f'acc_{model_label(model)}'] = round(mdf['correct'].mean(), 4) if len(mdf) > 0 else np.nan
                    rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f'  Saved summary: {out_path}')
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--outdir', default=str(EVAL_DIR / 'size_analysis'))
    parser.add_argument('--baseline', default='ptv3_baseline_120ep',
                        help='Model to use as baseline for delta plots')
    parser.add_argument('--split', default='test',
                        help='Which split predictions to analyse: test or val_ood')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Loading geometry...')
    geom_df = load_geometry()
    print(f'  {len(geom_df)} trees total')

    print('Loading predictions...')
    split_subdir = args.split if args.split != 'test' else ''
    pred_df = load_predictions(args.models, split_subdir=split_subdir)
    models_loaded = pred_df['model'].unique().tolist()
    print(f'  Models: {models_loaded}')

    print('Merging...')
    df = merge(geom_df, pred_df, geom_split=args.split)
    print(f'  Merged: {len(df)} rows')

    # Add size bins (computed on all test rows, model-agnostic)
    test_one_model = df[df['model'] == models_loaded[0]].copy()
    _, height_labels = add_bins(test_one_model, 'height')
    _, crown_labels = add_bins(test_one_model[~test_one_model['hull_degenerate']], 'crown_area')

    # Recompute bins on full df using same edges
    df, height_labels = add_bins(df, 'height')
    df_valid_hull = df[~df['hull_degenerate']].copy()
    df_valid_hull, crown_labels = add_bins(df_valid_hull, 'crown_area')

    print(f'  Height bins: {height_labels}')
    print(f'  Crown area bins: {crown_labels}')

    print('\nGenerating plots...')

    # 1. Accuracy vs. height bins per genus
    plot_accuracy_vs_bins(df, 'height_bin', height_labels, models_loaded,
                          'Per-genus', outdir / '1_acc_vs_height_by_genus.png')

    # 2. Accuracy vs. crown area bins per genus
    plot_accuracy_vs_bins(df_valid_hull, 'crown_area_bin', crown_labels, models_loaded,
                          'Per-genus', outdir / '2_acc_vs_crown_area_by_genus.png')

    # 3. Scatter: height vs crown area, errors
    plot_scatter_size_errors(df, models_loaded, outdir / '3_error_scatter_size.png')

    # 4. Size distributions per dataset
    plot_size_distributions(geom_df, outdir / '4_size_distribution_by_dataset.png')

    # 5. Delta accuracy vs. baseline by height bin
    if args.baseline in models_loaded:
        plot_delta_accuracy(df, 'height_bin', height_labels, models_loaded,
                            args.baseline, outdir / '5a_delta_acc_vs_height.png')
        plot_delta_accuracy(df_valid_hull, 'crown_area_bin', crown_labels, models_loaded,
                            args.baseline, outdir / '5b_delta_acc_vs_crown_area.png')

    # 6. Size distributions per genus
    plot_size_by_genus(geom_df, outdir / '6_size_distribution_by_genus.png')

    # 7. Accuracy vs. height bins per dataset
    plot_accuracy_vs_bins_by_dataset(df, 'height_bin', height_labels, models_loaded,
                                     'Per-dataset', outdir / '7a_acc_vs_height_by_dataset.png')
    plot_accuracy_vs_bins_by_dataset(df_valid_hull, 'crown_area_bin', crown_labels, models_loaded,
                                     'Per-dataset', outdir / '7b_acc_vs_crown_area_by_dataset.png')

    # Summary table — use df_valid_hull so crown_area_bin is present
    # propagate crown_area_bin back into df for rows that have it
    df = df.merge(df_valid_hull[['stem', 'model', 'crown_area_bin']],
                  on=['stem', 'model'], how='left')
    make_summary(df, models_loaded, outdir / 'summary.csv')

    print('\nDone.')


if __name__ == '__main__':
    main()
