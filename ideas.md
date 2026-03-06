# Fusion Strategy Ideas

## Current approach: Late concat
- Project each stream to 128d, concatenate, feed to MLP classifier
- Backbone never sees context — fusion happens after all processing is done
- Simple but the MLP applies fixed weights regardless of input

## Cross-attention fusion (modality tokens)
- Treat each modality as a token (or multiple tokens), run self-attention over them
- Attention weights are input-dependent — model learns when to trust which modality
- Can scale to more modalities/tokens without architectural changes
- Variants: 3 tokens (1 per modality), or many tokens (keep point patches from backbone bottleneck + context tokens)

## Mid-backbone injection
- Concatenate context features to per-point features at an intermediate PTv3 encoder stage
- Project back to original dimension so downstream stages are unchanged
- All subsequent transformer layers process context-aware point features
- Context is identical across points but gets mixed with each point's unique geometry after projection

### Injection point candidates (PTv3 encoder stages):
| Stage | Points | Feature dim | Context ratio (70d / feat_dim) |
|-------|--------|-------------|-------------------------------|
| 0 | 8192 | 32d | 219% — too dominant |
| 1 | 4096 | 64d | 109% — still too much |
| 2 | 2048 | 128d | 55% — reasonable |
| 3 | 1024 | 256d | 27% — comfortable |
| 4 | 512 | 512d | 14% — minimal impact |

Could try stages 2, 3, 4 and compare. Earlier injection = more processing with context but higher ratio risk.

## Whole plot as context stream
- Process the full unsegmented circular plot (15m radius TLS) through a separate backbone
- TLS is very dense — needs aggressive downsampling to fixed point count (e.g. 8192 or 16384)
- Captures stand structure, density, species mix geometry, ground/understory — 3D info that AE and BDL can't provide
- Separate (lighter?) backbone for the plot stream since the scale and structure differ from individual trees
- Fuse plot features with tree features via late concat or cross-attention tokens
- Border tree truncation is not an issue — the model sees the plot as-is
- Cross-attention between individual segmented trees was considered but rejected: variable tree count (5-100), truncation artifacts, batching complexity, label leakage risk

## Other ideas
- **PTv3 + BDL only (no AE)**: ablation to check if BDL adds anything without AE
- **FiLM conditioning**: use context to generate scale/shift params that modulate point features — lightweight alternative to concat injection
- **Ensemble**: train separate models per modality, combine predictions at test time — gives upper bound on complementarity
