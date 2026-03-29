"""
Summarize all dual-val experiment results into a CSV table.

Metrics extracted from the last evaluated epoch:
  - mAcc, allAcc from train.log (last occurrence of mIoU/mAcc/allAcc line)
  - weighted F1 computed from the saved confusion matrix (overwritten each epoch → last epoch)

Run:
  python tools/summarize_dual_val_results.py
"""

import os
import re
import numpy as np
import csv

EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Pointcept', 'exp', 'snapshot_10class_dual_val')
EXP_ROOT = os.path.realpath(EXP_ROOT)
OUT_CSV  = os.path.join(EXP_ROOT, '..', '..', '..', '..', 'results_dual_val_summary.csv')
OUT_CSV  = os.path.realpath(OUT_CSV)

# Human-readable architecture labels
ARCH_LABELS = {
    'smoke_test':                          'PTv3-baseline (smoke)',
    'ptv3_small_4gpu_50ep':               'PTv3-baseline',
    'ptv3_small_4gpu_100ep':              'PTv3-baseline',
    'ptv3_small_4gpu_300ep':              'PTv3-baseline',
    'ptv3_small_8gpu_300ep':              'PTv3-baseline',
    'ptv3_ctx_ae_4gpu_100ep':             'PTv3 + AE (concat)',
    'ptv3_ctx_ae_4gpu_100ep_vmf':         'PTv3 + AE (concat, vMF aug)',
    'ptv3_ctx_ae_8gpu_200ep':             'PTv3 + AE (concat)',
    'ptv3_ctx_sinr_4gpu_100ep':           'PTv3 + SINR (concat)',
    'ptv3_ctx_ae_sinr_cat_4gpu_200ep':    'PTv3 + AE + SINR (concat)',
    'ptv3_ctx_ae_sinr_cat_aux_4gpu_200ep':'PTv3 + AE + SINR (concat, aux heads)',
    'ptv3_ctx_ae_sinr_cat_aux_frozen_4gpu_50ep': 'PTv3-frozen + AE + SINR (concat, aux heads)',
    'ptv3_ctx_ae_sinr_xattn_aux_4gpu_200ep':     'PTv3 + AE + SINR (cross-attn, aux heads)',
}


def weighted_f1_from_cm(cm):
    """Compute weighted F1 from a (C, C) confusion matrix (rows=true, cols=pred)."""
    n_classes = cm.shape[0]
    f1_per_class = []
    support = cm.sum(axis=1)  # true positives + false negatives per class
    total = support.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)
    weighted_f1 = sum(f1_per_class[c] * support[c] for c in range(n_classes)) / total
    return weighted_f1


def get_last_epoch_from_log(log_path):
    """Return (last_epoch_completed, total_epochs_configured)."""
    last, total = None, None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Train: \[(\d+)/(\d+)\]', line)
            if m:
                last  = int(m.group(1))
                total = int(m.group(2))
    return last, total


def get_last_metrics_from_log(log_path, split):
    """Return (mAcc, allAcc) for the last evaluation of `split` in the log."""
    pattern = re.compile(
        rf'{split}: mIoU/mAcc/allAcc ([0-9.]+)/([0-9.]+)/([0-9.]+)'
    )
    m_acc = all_acc = None
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                m_acc    = float(m.group(2))
                all_acc  = float(m.group(3))
    return m_acc, all_acc


def process_run(run_dir):
    name     = os.path.basename(run_dir)
    log_path = os.path.join(run_dir, 'train.log')
    cm_id    = os.path.join(run_dir, 'confusion_matrix_val_id.npy')
    cm_ood   = os.path.join(run_dir, 'confusion_matrix_val_ood.npy')

    if not os.path.exists(log_path):
        return None

    arch = ARCH_LABELS.get(name, name)
    last_ep, total_ep = get_last_epoch_from_log(log_path)

    m_acc_id,  all_acc_id  = get_last_metrics_from_log(log_path, 'val_id')
    m_acc_ood, all_acc_ood = get_last_metrics_from_log(log_path, 'val_ood')

    # Skip runs with no completed evaluation
    if m_acc_id is None:
        return None

    wf1_id = wf1_ood = None
    if os.path.exists(cm_id):
        wf1_id  = weighted_f1_from_cm(np.load(cm_id))
    if os.path.exists(cm_ood):
        wf1_ood = weighted_f1_from_cm(np.load(cm_ood))

    # Show "last/total" if run didn't finish, else just last
    if last_ep is not None and total_ep is not None and last_ep < total_ep:
        ep_str = f'{last_ep}/{total_ep}'
    else:
        ep_str = str(last_ep) if last_ep else 'N/A'

    def fmt(v):
        return f'{v:.4f}' if v is not None else 'N/A'

    return {
        'run':           name,
        'arch':          arch,
        'epochs':        ep_str,
        'val_id_mAcc':   fmt(m_acc_id),
        'val_id_allAcc': fmt(all_acc_id),
        'val_id_wF1':    fmt(wf1_id),
        'val_ood_mAcc':  fmt(m_acc_ood),
        'val_ood_allAcc':fmt(all_acc_ood),
        'val_ood_wF1':   fmt(wf1_ood),
    }


COLUMNS = ['run', 'arch', 'epochs',
           'val_id_mAcc', 'val_id_allAcc', 'val_id_wF1',
           'val_ood_mAcc', 'val_ood_allAcc', 'val_ood_wF1']

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

# Print table
print(f"Saved to: {OUT_CSV}\n")
header = f"{'run':<45} {'arch':<40} {'ep':>4}  {'id_mAcc':>8} {'id_allAcc':>9} {'id_wF1':>8}  {'ood_mAcc':>8} {'ood_allAcc':>10} {'ood_wF1':>8}"
print(header)
print('-' * len(header))
for r in rows:
    print(f"{r['run']:<45} {r['arch']:<40} {str(r['epochs']):>4}  "
          f"{r['val_id_mAcc']:>8} {r['val_id_allAcc']:>9} {r['val_id_wF1']:>8}  "
          f"{r['val_ood_mAcc']:>8} {r['val_ood_allAcc']:>10} {r['val_ood_wF1']:>8}")
