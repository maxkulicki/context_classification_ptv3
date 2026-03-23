"""
Context Fusion Classifiers for tree species classification.

Phase 1 — CtxCls-v1m1:
    PTv3 backbone + single-source context encoder + late concat fusion.

    Point cloud → PTv3 → mean pool → tree_feat (backbone_embed_dim, e.g. 512)
    Context vector → context_encoder → ctx (context_embed_dim, 256)
    concat → (backbone_embed_dim + context_embed_dim)
    → Linear(fused_dim → 512) + BN + ReLU + Dropout(0.5)
    → Linear(512 → num_classes)

    The context_key parameter tells the model which key to read from the data dict
    (e.g. "ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"). This matches the keys
    produced by StandardizedDataset when context_sources is configured.

Phase 2 — UniversalCtxCls-v1m1:
    PTv3 backbone + CLS-attention multi-source context encoder + late concat fusion.
    All 4 sources are active simultaneously with per-sample source dropout during
    training. A curriculum linearly ramps p_all (prob of using all 4 sources) from
    0 to p_all_end over warmup_epochs. Per-source auxiliary classification heads
    ensure gradient flows to every encoder regardless of source dropout state.
    At inference, set model.eval_sources to a list of source names to restrict
    which sources are used (None = all sources).
"""

import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point

from .cls_context_encoder import CLSContextEncoder, _sample_source_mask


@MODELS.register_module("CtxCls-v1m1")
class ContextFusionClassifier(nn.Module):
    """PTv3 + single-source context encoder, late concat fusion classifier.

    Parameters
    ----------
    backbone : dict
        Config for the point cloud backbone (e.g. PT-v3m1).
    context_encoder : dict
        Config for the context encoder (e.g. AEEncoder, TopoEncoder, ...).
    criteria : list[dict]
        Loss function configs.
    num_classes : int
        Number of output classes.
    backbone_embed_dim : int
        Dimensionality of the pooled point cloud feature from the backbone.
    context_embed_dim : int
        Output dimensionality of the context encoder (256).
    context_key : str
        Key in the data dict containing the raw context features for this source.
        Convention: "ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn".
    """

    def __init__(
        self,
        backbone=None,
        context_encoder=None,
        criteria=None,
        num_classes=13,
        backbone_embed_dim=512,
        context_embed_dim=256,
        context_key="ctx_ae",
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.context_encoder = build_model(context_encoder)
        self.criteria = build_criteria(criteria)
        self.context_key = context_key

        fused_dim = backbone_embed_dim + context_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        # Global mean-pool point features: (N_total, C) → (B, C)
        if isinstance(point, Point):
            tree_feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
        else:
            tree_feat = point

        ctx_raw = input_dict[self.context_key]  # (B, raw_dim)
        ctx = self.context_encoder(ctx_raw)      # (B, context_embed_dim)

        fused = torch.cat([tree_feat, ctx], dim=-1)  # (B, fused_dim)
        cls_logits = self.cls_head(fused)

        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module("MultiCatCtxCls-v1m1")
class MultiCatCtxCls(nn.Module):
    """PTv3 + multiple context encoders, deterministic late concat fusion.

    No source dropout, no attention, no aux heads. All sources are always
    active. Encodes each source independently, concatenates all context
    embeddings with the pooled point cloud feature, then classifies.

        tree_feat (backbone_embed_dim)
        || ctx_0 (context_embed_dim)
        || ctx_1 (context_embed_dim)
        || ...
        → Linear(fused_dim, 512) → BN → ReLU → Dropout → Linear(512, num_classes)

    Parameters
    ----------
    backbone : dict
        Config for the point cloud backbone (e.g. PT-v3m1).
    context_encoders : list[dict]
        List of encoder configs, one per source.
    context_keys : list[str]
        Data dict keys for each source, same order as context_encoders.
        E.g. ["ctx_ae", "ctx_sinr"].
    criteria : list[dict]
        Loss function configs.
    num_classes : int
    backbone_embed_dim : int
    context_embed_dim : int
        Output dim of each encoder (all must share the same dim).
    """

    def __init__(
        self,
        backbone=None,
        context_encoders=None,
        context_keys=None,
        criteria=None,
        num_classes=13,
        backbone_embed_dim=512,
        context_embed_dim=256,
    ):
        super().__init__()
        if context_encoders is None:
            context_encoders = []
        if context_keys is None:
            context_keys = []
        assert len(context_encoders) == len(context_keys), \
            "context_encoders and context_keys must have the same length"

        self.backbone = build_model(backbone)
        self.context_encoders = nn.ModuleList(
            [build_model(cfg) for cfg in context_encoders]
        )
        self.context_keys = context_keys
        self.criteria = build_criteria(criteria)

        fused_dim = backbone_embed_dim + len(context_encoders) * context_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        tree_feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="mean",
        )

        ctx_feats = [
            enc(input_dict[key])
            for enc, key in zip(self.context_encoders, self.context_keys)
        ]
        fused = torch.cat([tree_feat] + ctx_feats, dim=-1)
        cls_logits = self.cls_head(fused)

        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module("UniversalCtxCls-v1m1")
class UniversalCtxCls(nn.Module):
    """PTv3 + CLS-attention multi-source context encoder, late concat fusion.

    All 4 context sources are encoded simultaneously. During training, a
    per-sample source dropout mask controls which sources participate in the
    CLS attention (all encoders still receive gradients via auxiliary heads).
    A curriculum linearly ramps p_all from 0 to source_dropout_prob over
    source_dropout_warmup epochs, preventing the CLS token from locking onto
    the strongest source early in training.

    At inference, all sources are active by default. Set model.eval_sources
    to a list of source names (e.g. ['ae', 'sinr']) before evaluation to
    restrict which sources are used — useful for the masking evaluation table.

    Parameters
    ----------
    backbone : dict
        Config for the PTv3 point cloud backbone.
    cls_context_encoder : dict
        Config for the CLSCtxEncoder-v1m1 multi-source aggregator.
    criteria : list[dict]
        Loss function configs for the fused classification head.
    num_classes : int
        Number of output classes.
    backbone_embed_dim : int
        Dimensionality of the pooled point cloud feature (512).
    context_embed_dim : int
        Output dimensionality of the context encoder (256).
    source_keys : list[str]
        Data dict keys for each source, in the order ['ae', 'topo', 'sinr', 'gpn'].
        Convention: ["ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"].
    source_dropout_prob : float
        Final probability of activating all 4 sources (target p_all = 0.35).
    source_dropout_warmup : int
        Epochs over which p_all ramps from 0 to source_dropout_prob.
    aux_loss_weight : float
        Weight for each per-source auxiliary cross-entropy loss (0.1).
    """

    # Must match CLSContextEncoder.SOURCE_ORDER
    SOURCE_NAMES = ["ae", "topo", "sinr", "gpn"]

    def __init__(
        self,
        backbone=None,
        cls_context_encoder=None,
        criteria=None,
        num_classes=13,
        backbone_embed_dim=512,
        context_embed_dim=256,
        source_keys=None,
        source_dropout_prob=0.35,
        source_dropout_warmup=40,
        aux_loss_weight=0.1,
    ):
        super().__init__()
        if source_keys is None:
            source_keys = ["ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"]

        self.backbone = build_model(backbone)
        self.cls_context_encoder = build_model(cls_context_encoder)
        self.criteria = build_criteria(criteria)
        self.source_keys = source_keys
        self.source_dropout_prob = source_dropout_prob
        self.source_dropout_warmup = source_dropout_warmup
        self.aux_loss_weight = aux_loss_weight

        # Auxiliary cross-entropy head per source (discarded at inference)
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(context_embed_dim, num_classes) for name in self.SOURCE_NAMES}
        )

        # Fused classification head: (512 + 256) → 512 → num_classes
        fused_dim = backbone_embed_dim + context_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # Set at inference time to restrict which sources are active.
        # None = all sources. Example: ['ae', 'sinr'] for AE+SINR only.
        self.eval_sources = None

    def forward(self, input_dict):
        # 1. PTv3 backbone + global mean pool → (B, 512)
        point = Point(input_dict)
        point = self.backbone(point)
        tree_feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="mean",
        )
        B = tree_feat.shape[0]

        # 2. Gather raw source tensors from data dict
        source_dict = {
            name: input_dict[key]
            for name, key in zip(self.SOURCE_NAMES, self.source_keys)
        }

        # 3. Determine source activation mask
        if self.training:
            # Curriculum source dropout: p_all ramps from 0 to target over warmup
            epoch = input_dict.get("epoch", None)
            active_mask = _sample_source_mask(
                B,
                tree_feat.device,
                epoch,
                p_all_end=self.source_dropout_prob,
                warmup_epochs=self.source_dropout_warmup,
            )
        elif self.eval_sources is not None:
            # Inference with a restricted source subset
            idx = [self.SOURCE_NAMES.index(s) for s in self.eval_sources]
            active_mask = torch.zeros(B, 4, dtype=torch.bool, device=tree_feat.device)
            active_mask[:, idx] = True
        else:
            active_mask = None  # all sources active

        # 4. CLS-attention context encoding
        #    ctx: (B, 256) — fused context embedding
        #    source_tokens: dict[str, (B, 256)] — per-source embeddings for aux heads
        ctx, source_tokens = self.cls_context_encoder(source_dict, active_mask)

        # 5. Late concat fusion + classification
        fused = torch.cat([tree_feat, ctx], dim=-1)  # (B, 768)
        cls_logits = self.cls_head(fused)

        # 6. Return logits only if no labels are available (test-time inference)
        if "category" not in input_dict:
            return dict(cls_logits=cls_logits)

        labels = input_dict["category"]
        loss_fused = self.criteria(cls_logits, labels)

        # Auxiliary heads: always applied to all source tokens to ensure each
        # encoder receives gradient even when its source is dropped from attention.
        aux_logits = {name: self.aux_heads[name](tok) for name, tok in source_tokens.items()}
        aux_losses = {name: self.aux_criterion(lgts, labels) for name, lgts in aux_logits.items()}
        loss_aux_total = sum(aux_losses.values())
        loss = loss_fused + self.aux_loss_weight * loss_aux_total

        if self.training:
            # avg active sources: mean number of active sources per sample this batch.
            # active_mask is (B, 4) bool; None means all 4 were active (eval fallback).
            if active_mask is not None:
                avg_active = active_mask.float().sum(dim=1).mean()
            else:
                avg_active = torch.tensor(4.0, device=tree_feat.device)
            return dict(
                loss=loss,
                loss_fused=loss_fused,
                loss_aux_ae=aux_losses["ae"],
                loss_aux_topo=aux_losses["topo"],
                loss_aux_sinr=aux_losses["sinr"],
                loss_aux_gpn=aux_losses["gpn"],
                avg_active_sources=avg_active,
            )
        # Eval with labels: return logits for both fused head and aux heads so the
        # evaluator can compute per-source auxiliary accuracy.
        return dict(
            loss=loss,
            cls_logits=cls_logits,
            aux_logits_ae=aux_logits["ae"],
            aux_logits_topo=aux_logits["topo"],
            aux_logits_sinr=aux_logits["sinr"],
            aux_logits_gpn=aux_logits["gpn"],
        )
