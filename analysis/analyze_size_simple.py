"""
Focused size vs. accuracy analysis — 3 clean plots.

Usage:
    conda run -n context_baseline python analyze_size_simple.py [--split test|val_ood|val_id]
                                                                 [--outdir ...]
                                                                 [--models ...]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

ROOT     = Path('/net/pr2/projects/plgrid/plggtreeseg')
EVAL_DIR = ROOT / 'context_classification_ptv3/Pointcept/eval_results'
GEOM_CSV = ROOT / 'context_classification_ptv3/data/tree_geometry.csv'

DEFAULT_MODELS = [
    'ptv3_baseline_120ep',
    'ptv3_sinr_gauss_120ep',
    'ptv3_ae_combined_aug_vmf_120ep',
    'ptv3_ae_sinr_cat_aux_200ep',
]
MODEL_LABELS = {
    'ptv3_baseline_120ep':              'Baseline',
    'ptv3_sinr_gauss_120ep':            'SINR',
    'ptv3_ae_combined_aug_vmf_120ep':   'AE',
    'ptv3_ae_sinr_cat_aux_200ep':       'AE+SINR',
}
MODEL_COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

GENERA = ['Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
          'Fagus', 'Larix', 'Picea', 'Pinus', 'Quercus']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(models, split):
    geom = pd.read_csv(GEOM_CSV)
    geom = geom[geom['split'] == split][['stem', 'height', 'crown_area', 'hull_degenerate']]

    subdir = '' if split == 'test' else split
    frames = []
    for model in models:
        p = (EVAL_DIR / subdir / model / 'predictions.csv') if subdir else \
            (EVAL_DIR / model / 'predictions.csv')
        if not p.exists():
            print(f'  WARNING: missing {p}')
            continue
        df = pd.read_csv(p)
        df['model'] = model
        frames.append(df)

    preds = pd.concat(frames, ignore_index=True)
    df = preds.merge(geom, on='stem', how='left')
    return df


# ---------------------------------------------------------------------------
# Plot 1 — Smooth accuracy vs. height per dataset
# ---------------------------------------------------------------------------

def plot_smooth_accuracy(df, models, window, out_path):
    datasets = sorted(df['dataset'].unique())
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

    for ax, ds in zip(axes, datasets):
        for model in models:
            sub = df[(df['dataset'] == ds) & (df['model'] == model)].dropna(subset=['height'])
            if len(sub) < window:
                continue
            sub = sub.sort_values('height')
            smoothed = sub['correct'].rolling(window, center=True, min_periods=window // 2).mean()
            ax.plot(sub['height'], smoothed,
                    color=colors[model], label=MODEL_LABELS.get(model, model),
                    linewidth=1.8, alpha=0.85)

        n = len(df[(df['dataset'] == ds) & (df['model'] == models[0])])
        ax.set_title(f'{ds}\n(n={n})', fontsize=10)
        ax.set_xlabel('Height (m)')
        ax.axhline(0.5, color='grey', lw=0.7, ls='--')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.25)

    axes[0].set_ylabel('Accuracy (rolling mean)')

    handles = [plt.Line2D([0], [0], color=colors[m], lw=2,
                           label=MODEL_LABELS.get(m, m)) for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=len(models),
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(f'Accuracy vs. tree height (rolling window={window}) — per dataset',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 2 — Height distribution: correct vs. wrong per genus
# ---------------------------------------------------------------------------

def plot_error_distributions(df, models, out_path):
    # Show the two most contrasting models to keep it readable
    m_base = models[0]
    m_ctx  = models[-1]
    pair   = [m_base, m_ctx]
    labels = [MODEL_LABELS.get(m, m) for m in pair]
    colors_ok   = ['#4878CF', '#B47CC7']
    colors_err  = ['#9BB8E8', '#DCBEF0']

    genera_present = [g for g in GENERA if g in df['true_genus'].unique()]
    ncols = 5
    nrows = int(np.ceil(len(genera_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 3.2))
    axes = axes.flatten()

    for i, genus in enumerate(genera_present):
        ax = axes[i]
        gdf = df[df['true_genus'] == genus].dropna(subset=['height'])

        positions = []
        data_list = []
        tick_labels = []
        face_colors = []

        x = 0
        for j, model in enumerate(pair):
            mdf = gdf[gdf['model'] == model]
            correct = mdf[mdf['correct'] == 1]['height'].values
            wrong   = mdf[mdf['correct'] == 0]['height'].values

            for vals, label_suffix, fc in [
                (correct, '✓', colors_ok[j]),
                (wrong,   '✗', colors_err[j]),
            ]:
                if len(vals) >= 3:
                    vp = ax.violinplot([vals], positions=[x], widths=0.7,
                                       showmedians=True, showextrema=False)
                    for pc in vp['bodies']:
                        pc.set_facecolor(fc)
                        pc.set_alpha(0.75)
                    vp['cmedians'].set_color('black')
                    vp['cmedians'].set_linewidth(1.2)
                else:
                    ax.scatter([x] * len(vals), vals, color=fc, s=15, alpha=0.6)
                positions.append(x)
                tick_labels.append(f'{labels[j]}\n{label_suffix}')
                x += 1
            x += 0.4  # gap between models

        n = len(gdf) // len(pair)
        ax.set_title(f'{genus} (n={n})', fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=6.5)
        ax.set_ylabel('Height (m)' if i % ncols == 0 else '')
        ax.grid(axis='y', alpha=0.2)

    for i in range(len(genera_present), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f'Height distribution: correctly vs. incorrectly classified trees per genus\n'
        f'({labels[0]} vs. {labels[1]})',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 3 — Correlation heatmap: accuracy vs. size per model × dataset
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, models, out_path):
    datasets = sorted(df['dataset'].unique())
    row_labels = ['ALL'] + datasets
    col_labels  = [MODEL_LABELS.get(m, m) for m in models]

    def corr_matrix(size_col):
        mat = np.full((len(row_labels), len(models)), np.nan)
        sub = df.dropna(subset=[size_col])
        for j, model in enumerate(models):
            mdf = sub[sub['model'] == model]
            for i, row in enumerate(row_labels):
                rows = mdf if row == 'ALL' else mdf[mdf['dataset'] == row]
                if len(rows) < 10:
                    continue
                r, p = pointbiserialr(rows['correct'].values, rows[size_col].values)
                mat[i, j] = r
        return mat

    mat_h  = corr_matrix('height')
    mat_ca = corr_matrix('crown_area')

    fig, axes = plt.subplots(1, 2, figsize=(len(models) * 2 + 2, len(row_labels) * 0.7 + 1.5))
    vmax = max(np.nanmax(np.abs(mat_h)), np.nanmax(np.abs(mat_ca)), 0.01)

    for ax, mat, title in zip(axes, [mat_h, mat_ca], ['Height', 'Crown area']):
        im = ax.imshow(mat, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='r')
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(models)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            fontsize=8,
                            color='white' if abs(v) > vmax * 0.6 else 'black')
        # Separator line after ALL row
        ax.axhline(0.5, color='black', lw=1.2)

    fig.suptitle('Point-biserial correlation: accuracy ~ tree size\n'
                 'Positive = larger trees classified more accurately', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',  nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--split',   default='test')
    parser.add_argument('--outdir',  default=str(EVAL_DIR / 'size_analysis_v2'))
    parser.add_argument('--window',  type=int, default=60,
                        help='Rolling window size for Plot 1')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data (split={args.split})...')
    df = load_data(args.models, args.split)
    models_loaded = df['model'].unique().tolist()
    print(f'  {len(df) // len(models_loaded)} trees, {len(models_loaded)} models')

    print('\nPlot 1: smooth accuracy vs. height...')
    plot_smooth_accuracy(df, models_loaded, args.window,
                         outdir / '1_smooth_acc_vs_height.png')

    print('Plot 2: correct vs. wrong height distributions per genus...')
    plot_error_distributions(df, models_loaded,
                             outdir / '2_height_correct_vs_wrong_per_genus.png')

    print('Plot 3: correlation heatmap...')
    plot_correlation_heatmap(df, models_loaded,
                             outdir / '3_correlation_heatmap.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
