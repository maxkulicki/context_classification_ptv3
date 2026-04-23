"""
Merge ctx_sinr from context_features.pth into context_features_combined_aug.pth.

Output: context_features_combined_aug_sinr.pth
  per stem: {"ctx_ae": (64,), "ctx_ae_pool": (50, 64), "ctx_sinr": (256,)}

Usage:
    conda run -n context_baseline python merge_context_features_combined_aug_sinr.py
"""

from pathlib import Path
import torch

DATA_DIR = Path('/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/data/snapshot_v1')
COMBINED_PTH = DATA_DIR / 'context_features_combined_aug.pth'
SINR_PTH     = DATA_DIR / 'context_features.pth'
OUT_PTH      = DATA_DIR / 'context_features_combined_aug_sinr.pth'

print(f'Loading {COMBINED_PTH} ...')
combined = torch.load(COMBINED_PTH, map_location='cpu', weights_only=False)
print(f'  {len(combined)} stems')

print(f'Loading {SINR_PTH} ...')
sinr_src = torch.load(SINR_PTH, map_location='cpu', weights_only=False)
print(f'  {len(sinr_src)} stems')

missing = 0
for stem, entry in combined.items():
    if stem in sinr_src and 'ctx_sinr' in sinr_src[stem]:
        entry['ctx_sinr'] = sinr_src[stem]['ctx_sinr']
    else:
        missing += 1

if missing:
    print(f'WARNING: {missing} stems had no ctx_sinr in source — they will be absent from output')
else:
    print('All stems merged successfully.')

# Verify first entry
first_stem = next(iter(combined))
print(f'Keys in first stem ({first_stem}): {list(combined[first_stem].keys())}')

print(f'Saving to {OUT_PTH} ...')
torch.save(combined, OUT_PTH)
print('Done.')
