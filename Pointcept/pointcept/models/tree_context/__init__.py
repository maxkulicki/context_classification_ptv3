"""
Tree context fusion module for multimodal species classification.

Structure (mirrors experiment phases):
    encoders.py             — Per-source context encoders (Phases 1–4)
    classifier.py           — CtxCls-v1m1: single-source late fusion (Phase 1)
                            — UniversalCtxCls-v1m1: multi-source CLS fusion (Phase 2)
    cls_context_encoder.py  — CLS-attention multi-source aggregator (Phase 2)

Planned additions:
    film_classifier.py      — Mid-fusion via FiLM conditioning (Phase 3)
    inter_tree_attention.py — Neighborhood-level reasoning (Phase 4)
"""

from .encoders import (
    AlphaEarthEncoder,
    TopoEncoder,
    SINREncoder,
    GeoPlantNetEncoder,
)
from .cls_context_encoder import CLSContextEncoder
from .classifier import ContextFusionClassifier, MultiCatCtxCls, UniversalCtxCls

__all__ = [
    "AlphaEarthEncoder",
    "TopoEncoder",
    "SINREncoder",
    "GeoPlantNetEncoder",
    "CLSContextEncoder",
    "ContextFusionClassifier",
    "MultiCatCtxCls",
    "UniversalCtxCls",
]
