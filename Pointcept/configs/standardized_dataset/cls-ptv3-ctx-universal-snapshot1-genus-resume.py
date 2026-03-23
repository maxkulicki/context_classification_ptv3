_base_ = ["../_base_/default_runtime.py"]

# Phase 2, Experiment 2.0 — Continuation run (epochs 50 → 99)
# Resumes from model_last.pth; epoch index continues from checkpoint.
# Fresh OneCycleLR for 100 epochs (we run the second half: 50–99).
# Optimizer/scheduler state NOT restored — fresh schedule from checkpoint weights.

# misc custom setting
batch_size = 128  # total across all GPUs (16 per GPU × 8 GPUs)
num_worker = 32
batch_size_val = 256
empty_cache = False
enable_amp = True

EXP_DIR = "/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/Pointcept/exp/snapshot1/ptv3_ctx_universal_8gpu_100ep"

weight = EXP_DIR + "/model/model_last.pth"
resume = True

# model settings (identical to original run)
model = dict(
    type="UniversalCtxCls-v1m1",
    num_classes=13,
    backbone_embed_dim=512,
    context_embed_dim=256,
    source_keys=["ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"],
    source_dropout_prob=0.35,
    source_dropout_warmup=40,
    aux_loss_weight=0.1,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    cls_context_encoder=dict(
        type="CLSCtxEncoder-v1m1",
        embed_dim=256,
        num_heads=4,
        ae_cfg=dict(type="AEEncoder", input_dim=64, output_dim=256),
        topo_cfg=dict(type="TopoEncoder", input_dim=6, hidden_dim=128, output_dim=256),
        sinr_cfg=dict(type="SINREncoder", input_dim=256, output_dim=256),
        gpn_cfg=dict(type="GPNEncoder", input_dim=18, output_dim=256),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
# eval_epoch=100 → max_epoch=100; with start_epoch=50 from checkpoint,
# we run 50 more epochs. OneCycleLR total_steps = N*100; we consume N*50 steps
# (the second half of the cycle: peak → cosine decay to near zero).
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="backbone", lr=0.0001)]

# dataset settings (identical to original run)
dataset_type = "StandardizedDataset"
data_root = "/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_1_npy_fps8192"
context_pth = "/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/data/snapshot_v1/context_features.pth"
cache_data = False
class_names = [
    "Abies",
    "Acer",
    "Alnus",
    "Betula",
    "Carpinus",
    "Fagus",
    "Fraxinus",
    "Larix",
    "Picea",
    "Pinus",
    "Pseudotsuga",
    "Quercus",
    "Tilia",
]

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        label_level="genus",
        context_pth=context_pth,
        context_sources=["alphaearth", "topo", "sinr", "gpn"],
        transform=[
            dict(type="CenterShiftMean"),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category", "ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        label_level="genus",
        context_pth=context_pth,
        context_sources=["alphaearth", "topo", "sinr", "gpn"],
        transform=[
            dict(type="CenterShiftMean"),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category", "ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        label_level="genus",
        context_pth=context_pth,
        context_sources=["alphaearth", "topo", "sinr", "gpn"],
        transform=[
            dict(type="CenterShiftMean"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "ctx_ae", "ctx_topo", "ctx_sinr", "ctx_gpn"),
                    feat_keys=["coord"],
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
            ],
        ),
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader", keywords="", replacement="", strict=False,
         resume_optimizer=False, resume_scheduler=False),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="UniversalClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

# tester
test = dict(type="ClsVotingTester", num_repeat=10)
