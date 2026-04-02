"""
Extended summary of dual-val experiment results into a CSV table.

Metrics extracted from the last evaluated epoch:
  - mIoU, mAcc, allAcc from train.log
  - weighted F1 computed from the saved confusion matrix
  - per-class IoU and Acc from train.log
  - per-class precision, recall, F1 from confusion matrix
  - per-dataset accuracy from train.log

Run:
  python tools/summarize_dual_val_extended.py
"""

import os
import re
import numpy as np
import csv

EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Pointcept', 'exp', 'snapshot_10class_dual_val')
EXP_ROOT = os.path.realpath(EXP_ROOT)
OUT_CSV  = os.path.join(EXP_ROOT, '..', '..', '..', '..', 'results_dual_val_extended.csv')
OUT_CSV  = os.path.realpath(OUT_CSV)

ARCH_LABELS = {
    'smoke_test':                                    'PTv3-baseline (smoke)',
    'ptv3_small_4gpu_50ep':                         'PTv3-baseline',
    'ptv3_small_4gpu_100ep':                        'PTv3-baseline',
    'ptv3_small_4gpu_300ep':                        'PTv3-baseline',
    'ptv3_small_8gpu_300ep':                        'PTv3-baseline',
    'ptv3_ctx_ae_4gpu_100ep':                       'PTv3 + AE (concat)',
    'ptv3_ctx_ae_4gpu_100ep_vmf':                   'PTv3 + AE (concat, vMF aug)',
    'ptv3_ctx_ae_8gpu_200ep':                       'PTv3 + AE (concat)',
    'ptv3_ctx_sinr_4gpu_100ep':                     'PTv3 + SINR (concat)',
    'ptv3_ctx_ae_sinr_cat_4gpu_200ep':              'PTv3 + AE + SINR (concat)',
    'ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep':          'PTv3 + AE + SINR (concat, aux heads)',
    'ptv3_ctx_ae_sinr_cat_aux_frozen_4gpu_50ep':    'PTv3-frozen + AE + SINR (concat, aux heads)',
    'ptv3_ctx_ae_sinr_xattn_aux_4gpu_200ep':        'PTv3 + AE + SINR (cross-attn, aux heads)',
    'ptv3_ctx_ae_midfusion_4gpu_120ep':             'PTv3 + AE (mid-fusion)',
    'ptv3_ctx_ae_midfusion_stage2_4gpu_120ep':      'PTv3 + AE (mid-fusion, stage2)',
}

CLASSES = ['Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
           'Fagus', 'Larix', 'Picea', 'Pinus', 'Quercus']

DATASETS_ID  = ['CULS', 'Frey2022', 'Junttila', 'PulitiULS2',
                'Saarinen2021', 'TreeScanPL', 'Weiser']
DATASETS_OOD = ['CULS', 'Frey2022', 'Junttila',
                'Saarinen2021', 'TreeScanPL', 'Weiser']


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_last_epoch_from_log(log_path):
    last, total = None, None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Train: \[(\d+)/(\d+)\]', line)
            if m:
                last  = int(m.group(1))
                total = int(m.group(2))
    return last, total


def get_last_summary_from_log(log_path, split):
    """Return (mIoU, mAcc, allAcc) for the last evaluation of `split`."""
    pattern = re.compile(
        rf'{split}: mIoU/mAcc/allAcc ([0-9.]+)/([0-9.]+)/([0-9.]+)'
    )
    miou = macc = allacc = None
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                miou   = float(m.group(1))
                macc   = float(m.group(2))
                allacc = float(m.group(3))
    return miou, macc, allacc


def get_last_class_metrics_from_log(log_path, split):
    """Return dict {class_name: (iou, acc)} from last occurrence of each class."""
    pattern = re.compile(
        rf'{split} Class_\d+-(\w+): iou/acc ([0-9.]+)/([0-9.]+)'
    )
    # Use dict so last occurrence wins
    result = {}
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                cls = m.group(1)
                result[cls] = (float(m.group(2)), float(m.group(3)))
    return result


def get_last_dataset_metrics_from_log(log_path, split):
    """Return dict {dataset_name: acc} from last occurrence of each dataset."""
    pattern = re.compile(
        rf'{split} dataset (\w+): acc ([0-9.]+)'
    )
    result = {}
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                result[m.group(1)] = float(m.group(2))
    return result


def per_class_cm_metrics(cm):
    """Return (precision, recall, f1) arrays of shape (C,) from confusion matrix."""
    n = cm.shape[0]
    prec = np.zeros(n)
    rec  = np.zeros(n)
    f1   = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c]   = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                   if (prec[c] + rec[c]) > 0 else 0.0)
    return prec, rec, f1


def weighted_f1_from_cm(cm):
    _, _, f1 = per_class_cm_metrics(cm)
    support = cm.sum(axis=1)
    return float((f1 * support).sum() / support.sum())


# ---------------------------------------------------------------------------
# Build one row per run
# ---------------------------------------------------------------------------

def fmt(v):
    return f'{v:.4f}' if v is not None else 'N/A'


def process_run(run_dir):
    name     = os.path.basename(run_dir)
    log_path = os.path.join(run_dir, 'train.log')
    cm_id    = os.path.join(run_dir, 'confusion_matrix_val_id.npy')
    cm_ood   = os.path.join(run_dir, 'confusion_matrix_val_ood.npy')

    if not os.path.exists(log_path):
        return None

    arch = ARCH_LABELS.get(name, name)
    last_ep, total_ep = get_last_epoch_from_log(log_path)

    miou_id,  macc_id,  allacc_id  = get_last_summary_from_log(log_path, 'val_id')
    miou_ood, macc_ood, allacc_ood = get_last_summary_from_log(log_path, 'val_ood')

    if macc_id is None:
        return None

    cls_id  = get_last_class_metrics_from_log(log_path, 'val_id')
    cls_ood = get_last_class_metrics_from_log(log_path, 'val_ood')
    ds_id   = get_last_dataset_metrics_from_log(log_path, 'val_id')
    ds_ood  = get_last_dataset_metrics_from_log(log_path, 'val_ood')

    # Confusion-matrix metrics
    wf1_id = wf1_ood = None
    prec_id = rec_id = f1_id = None
    prec_ood = rec_ood = f1_ood = None
    if os.path.exists(cm_id):
        cm = np.load(cm_id)
        wf1_id = weighted_f1_from_cm(cm)
        prec_id, rec_id, f1_id = per_class_cm_metrics(cm)
    if os.path.exists(cm_ood):
        cm = np.load(cm_ood)
        wf1_ood = weighted_f1_from_cm(cm)
        prec_ood, rec_ood, f1_ood = per_class_cm_metrics(cm)

    ep_str = (f'{last_ep}/{total_ep}'
              if last_ep is not None and total_ep is not None and last_ep < total_ep
              else str(last_ep) if last_ep else 'N/A')

    row = {
        'run':   name,
        'arch':  arch,
        'epochs': ep_str,
        # Summary
        'val_id_mIoU':    fmt(miou_id),
        'val_id_mAcc':    fmt(macc_id),
        'val_id_allAcc':  fmt(allacc_id),
        'val_id_wF1':     fmt(wf1_id),
        'val_ood_mIoU':   fmt(miou_ood),
        'val_ood_mAcc':   fmt(macc_ood),
        'val_ood_allAcc': fmt(allacc_ood),
        'val_ood_wF1':    fmt(wf1_ood),
    }

    # Per-class IoU/Acc from log
    for cls in CLASSES:
        iou_id, acc_id   = cls_id.get(cls,  (None, None))
        iou_ood, acc_ood = cls_ood.get(cls, (None, None))
        row[f'val_id_iou_{cls}']  = fmt(iou_id)
        row[f'val_id_acc_{cls}']  = fmt(acc_id)
        row[f'val_ood_iou_{cls}'] = fmt(iou_ood)
        row[f'val_ood_acc_{cls}'] = fmt(acc_ood)

    # Per-class precision / F1 from confusion matrix
    for i, cls in enumerate(CLASSES):
        row[f'val_id_prec_{cls}']  = fmt(prec_id[i]  if prec_id  is not None else None)
        row[f'val_id_f1_{cls}']    = fmt(f1_id[i]    if f1_id    is not None else None)
        row[f'val_ood_prec_{cls}'] = fmt(prec_ood[i] if prec_ood is not None else None)
        row[f'val_ood_f1_{cls}']   = fmt(f1_ood[i]   if f1_ood   is not None else None)

    # Per-dataset accuracy
    for ds in DATASETS_ID:
        row[f'val_id_ds_{ds}'] = fmt(ds_id.get(ds))
    for ds in DATASETS_OOD:
        row[f'val_ood_ds_{ds}'] = fmt(ds_ood.get(ds))

    return row


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

COLUMNS = (
    ['run', 'arch', 'epochs']
    + ['val_id_mIoU', 'val_id_mAcc', 'val_id_allAcc', 'val_id_wF1']
    + ['val_ood_mIoU', 'val_ood_mAcc', 'val_ood_allAcc', 'val_ood_wF1']
    + [f'val_id_iou_{c}'  for c in CLASSES]
    + [f'val_id_acc_{c}'  for c in CLASSES]
    + [f'val_id_prec_{c}' for c in CLASSES]
    + [f'val_id_f1_{c}'   for c in CLASSES]
    + [f'val_ood_iou_{c}'  for c in CLASSES]
    + [f'val_ood_acc_{c}'  for c in CLASSES]
    + [f'val_ood_prec_{c}' for c in CLASSES]
    + [f'val_ood_f1_{c}'   for c in CLASSES]
    + [f'val_id_ds_{ds}'  for ds in DATASETS_ID]
    + [f'val_ood_ds_{ds}' for ds in DATASETS_OOD]
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

runs = sorted(os.listdir(EXP_ROOT))
rows = []
for run_name in runs:
    run_dir = os.path.join(EXP_ROOT, run_name)
    if not os.path.isdir(run_dir) or run_name == 'smoke_test':
        continue
    row = process_run(run_dir)
    if row:
        rows.append(row)

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} runs → {OUT_CSV}")
print(f"Columns: {len(COLUMNS)}")
print()

# Compact summary table
header = (f"{'run':<50} {'ep':>6}  "
          f"{'id_mIoU':>7} {'id_mAcc':>7} {'id_allAcc':>9} {'id_wF1':>7}  "
          f"{'ood_mIoU':>8} {'ood_mAcc':>8} {'ood_allAcc':>10} {'ood_wF1':>8}")
print(header)
print('-' * len(header))
for r in rows:
    print(f"{r['run']:<50} {str(r['epochs']):>6}  "
          f"{r['val_id_mIoU']:>7} {r['val_id_mAcc']:>7} {r['val_id_allAcc']:>9} {r['val_id_wF1']:>7}  "
          f"{r['val_ood_mIoU']:>8} {r['val_ood_mAcc']:>8} {r['val_ood_allAcc']:>10} {r['val_ood_wF1']:>8}")
