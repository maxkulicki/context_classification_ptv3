"""
Visualisation plots for dual-val experiment results.

Reads results_dual_val_extended.csv and produces:
  1. bar_val_ood_overall.png   – val_ood  mAcc / allAcc per run
  2. bar_val_id_overall.png    – val_id   mAcc / allAcc per run
  3. bar_id_vs_ood_mAcc.png    – val_id mAcc vs val_ood mAcc side-by-side
  4. bar_id_vs_ood_mIoU.png    – val_id mIoU vs val_ood mIoU side-by-side
  5. heatmap_class_iou_ood.png – per-class IoU (val_ood) across runs
  6. heatmap_class_iou_id.png  – per-class IoU (val_id) across runs
  7. heatmap_dataset_acc_ood.png – per-dataset accuracy (val_ood) across runs
  8. scatter_id_vs_ood.png     – val_id mAcc vs val_ood mAcc (generalisation gap)

Run:
  python tools/plot_dual_val_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Pointcept', 'exp', 'snapshot_10class_dual_val')
EXP_ROOT = os.path.realpath(EXP_ROOT)
CSV_PATH = os.path.join(EXP_ROOT, '..', '..', '..', '..', 'results_dual_val_extended.csv')
CSV_PATH = os.path.realpath(CSV_PATH)
OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH), 'plots_dual_val')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Short labels for axes (run_name → label)
# ---------------------------------------------------------------------------
SHORT_LABELS = {
    'ptv3_small_4gpu_50ep':                        'PTv3-50ep',
    'ptv3_small_4gpu_100ep':                       'PTv3-100ep',
    'ptv3_small_4gpu_300ep':                       'PTv3-300ep',
    'ptv3_small_8gpu_300ep':                       'PTv3-300ep-8g',
    'ptv3_ctx_ae_4gpu_100ep':                      'AE-cat-100',
    'ptv3_ctx_ae_4gpu_100ep_vmf':                  'AE-cat-100vMF',
    'ptv3_ctx_ae_8gpu_200ep':                      'AE-cat-200',
    'ptv3_ctx_sinr_4gpu_100ep':                    'SINR-cat',
    'ptv3_ctx_ae_sinr_cat_4gpu_200ep':             'AE+SINR-cat',
    'ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep':         'AE+SINR-cat-aux',
    'ptv3_ctx_ae_sinr_cat_aux_frozen_4gpu_50ep':   'AE+SINR-frz',
    'ptv3_ctx_ae_sinr_xattn_aux_4gpu_200ep':       'AE+SINR-xattn',
    'ptv3_ctx_ae_midfusion_4gpu_120ep':            'AE-mid',
    'ptv3_ctx_ae_midfusion_stage2_4gpu_120ep':     'AE-mid-s2',
}

CLASSES = ['Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
           'Fagus', 'Larix', 'Picea', 'Pinus', 'Quercus']
DATASETS_OOD = ['CULS', 'Frey2022', 'Junttila',
                'Saarinen2021', 'TreeScanPL', 'Weiser']

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
EXCLUDE_RUNS = {
    'ptv3_small_4gpu_50ep',
    'ptv3_small_8gpu_300ep',

    

}
df = df[~df['run'].isin(EXCLUDE_RUNS)]
df = df.replace('N/A', np.nan)

# Numeric conversion
num_cols = [c for c in df.columns if c not in ('run', 'arch', 'epochs')]
df[num_cols] = df[num_cols].astype(float)
df = df.copy()  # defragment after many column insertions from CSV

# Short labels column
df['label'] = df['run'].map(SHORT_LABELS).fillna(df['run'])

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
PALETTE   = sns.color_palette('tab10')
COLOR_ID  = PALETTE[0]   # blue
COLOR_OOD = PALETTE[1]   # orange
FIG_DPI   = 150


def sort_by(frame, cols):
    """Return a copy of frame sorted descending by the row-mean of `cols`."""
    key = frame[cols].mean(axis=1)
    return frame.loc[key.sort_values(ascending=False).index].reset_index(drop=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path}')


def bar_plot(ax, labels, values_list, bar_labels, colors, title, ylabel='Accuracy'):
    """Grouped bar chart on ax."""
    n_groups = len(labels)
    n_bars   = len(values_list)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)
    for i, (vals, lbl, col) in enumerate(zip(values_list, bar_labels, colors)):
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=lbl, color=col, alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    all_vals = [v for vals in values_list for v in vals if not np.isnan(v)]
    margin = (max(all_vals) - min(all_vals)) * 0.15
    ax.set_ylim(min(all_vals) - margin, min(max(all_vals) + margin * 3, 1.0))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. val_ood overall metrics
# ---------------------------------------------------------------------------
print('Generating plots...')

d = sort_by(df, ['val_ood_mAcc', 'val_ood_allAcc'])
fig, ax = plt.subplots(figsize=(14, 5))
bar_plot(
    ax,
    labels=d['label'].tolist(),
    values_list=[d['val_ood_mAcc'].tolist(), d['val_ood_allAcc'].tolist()],
    bar_labels=['mAcc', 'allAcc'],
    colors=[PALETTE[0], PALETTE[2]],
    title='val_ood — mAcc and allAcc per run',
)
save(fig, 'bar_val_ood_overall.png')

# ---------------------------------------------------------------------------
# 2. val_id overall metrics
# ---------------------------------------------------------------------------
d = sort_by(df, ['val_id_mAcc', 'val_id_allAcc'])
fig, ax = plt.subplots(figsize=(14, 5))
bar_plot(
    ax,
    labels=d['label'].tolist(),
    values_list=[d['val_id_mAcc'].tolist(), d['val_id_allAcc'].tolist()],
    bar_labels=['mAcc', 'allAcc'],
    colors=[PALETTE[0], PALETTE[2]],
    title='val_id — mAcc and allAcc per run',
)
save(fig, 'bar_val_id_overall.png')

# ---------------------------------------------------------------------------
# 3. ID vs OOD mAcc
# ---------------------------------------------------------------------------
d = sort_by(df, ['val_id_mAcc', 'val_ood_mAcc'])
fig, ax = plt.subplots(figsize=(14, 5))
bar_plot(
    ax,
    labels=d['label'].tolist(),
    values_list=[d['val_id_mAcc'].tolist(), d['val_ood_mAcc'].tolist()],
    bar_labels=['val_id mAcc', 'val_ood mAcc'],
    colors=[COLOR_ID, COLOR_OOD],
    title='ID vs OOD — mAcc per run',
)
save(fig, 'bar_id_vs_ood_mAcc.png')

# ---------------------------------------------------------------------------
# 4. ID vs OOD mIoU
# ---------------------------------------------------------------------------
d = sort_by(df, ['val_id_mIoU', 'val_ood_mIoU'])
fig, ax = plt.subplots(figsize=(14, 5))
bar_plot(
    ax,
    labels=d['label'].tolist(),
    values_list=[d['val_id_mIoU'].tolist(), d['val_ood_mIoU'].tolist()],
    bar_labels=['val_id mIoU', 'val_ood mIoU'],
    colors=[COLOR_ID, COLOR_OOD],
    title='ID vs OOD — mIoU per run',
    ylabel='mIoU',
)
save(fig, 'bar_id_vs_ood_mIoU.png')

# ---------------------------------------------------------------------------
# 5. Heatmap: per-class IoU — val_ood
# ---------------------------------------------------------------------------
iou_ood_cols = [f'val_ood_iou_{c}' for c in CLASSES]
d = sort_by(df, iou_ood_cols)
heat_ood = d[iou_ood_cols].values.astype(float)

fig, ax = plt.subplots(figsize=(13, max(4, len(d) * 0.55)))
sns.heatmap(
    heat_ood,
    ax=ax,
    xticklabels=CLASSES,
    yticklabels=d['label'].tolist(),
    annot=True, fmt='.3f', annot_kws={'size': 7},
    cmap='YlOrRd', vmin=0, vmax=1,
    linewidths=0.3, linecolor='white',
)
ax.set_title('val_ood — per-class IoU', fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout()
save(fig, 'heatmap_class_iou_ood.png')

# ---------------------------------------------------------------------------
# 6. Heatmap: per-class IoU — val_id
# ---------------------------------------------------------------------------
iou_id_cols = [f'val_id_iou_{c}' for c in CLASSES]
d = sort_by(df, iou_id_cols)
heat_id = d[iou_id_cols].values.astype(float)

fig, ax = plt.subplots(figsize=(13, max(4, len(d) * 0.55)))
sns.heatmap(
    heat_id,
    ax=ax,
    xticklabels=CLASSES,
    yticklabels=d['label'].tolist(),
    annot=True, fmt='.3f', annot_kws={'size': 7},
    cmap='YlOrRd', vmin=0, vmax=1,
    linewidths=0.3, linecolor='white',
)
ax.set_title('val_id — per-class IoU', fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout()
save(fig, 'heatmap_class_iou_id.png')

# ---------------------------------------------------------------------------
# 7. Heatmap: per-dataset accuracy — val_ood
# ---------------------------------------------------------------------------
ds_ood_cols = [f'val_ood_ds_{ds}' for ds in DATASETS_OOD]
d = sort_by(df, ds_ood_cols)
heat_ds = d[ds_ood_cols].values.astype(float)

fig, ax = plt.subplots(figsize=(10, max(4, len(d) * 0.55)))
sns.heatmap(
    heat_ds,
    ax=ax,
    xticklabels=DATASETS_OOD,
    yticklabels=d['label'].tolist(),
    annot=True, fmt='.3f', annot_kws={'size': 8},
    cmap='YlOrRd', vmin=0, vmax=1,
    linewidths=0.3, linecolor='white',
)
ax.set_title('val_ood — per-dataset accuracy', fontsize=12)
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout()
save(fig, 'heatmap_dataset_acc_ood.png')

# ---------------------------------------------------------------------------
# 8. Scatter: val_id mAcc vs val_ood mAcc (generalisation gap)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
x_vals = df['val_id_mAcc'].values
y_vals = df['val_ood_mAcc'].values
labels = df['label'].tolist()

colors_scatter = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
ax.scatter(x_vals, y_vals, s=80, c=colors_scatter, zorder=3)

for xi, yi, lbl in zip(x_vals, y_vals, labels):
    if not (np.isnan(xi) or np.isnan(yi)):
        ax.annotate(lbl, (xi, yi), textcoords='offset points',
                    xytext=(5, 3), fontsize=7, alpha=0.9)

# Diagonal reference line (perfect correlation)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='grey', alpha=0.5, linewidth=1, label='y = x')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('val_id mAcc', fontsize=10)
ax.set_ylabel('val_ood mAcc', fontsize=10)
ax.set_title('ID vs OOD generalisation (mAcc)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, 'scatter_id_vs_ood.png')

print(f'\nAll plots saved to: {OUT_DIR}')
