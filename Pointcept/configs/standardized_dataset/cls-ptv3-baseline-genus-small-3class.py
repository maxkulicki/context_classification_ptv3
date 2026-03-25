"""
PTv3 Small - Tree genus classification (3 classes: Betula, Picea, Pinus)
Subset of snapshot_v1 with stratified random 80/20 split (no location holdout).
Architecture identical to cls-ptv3-baseline-genus-small.py.
"""

weight = None
resume = False
evaluate = True
test_only = False
seed = 3313067
save_path = '/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/Pointcept/exp/snapshot_3class_mixed/ptv3_small_4gpu_100ep'
num_worker = 32
batch_size = 256
gradient_accumulation_steps = 1
batch_size_val = 512
batch_size_test = None
epoch = 50
eval_epoch = 50
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0
param_dicts = [dict(keyword='block', lr=0.0006)]

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='ClsEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
]

train = dict(type='DefaultTrainer')

model = dict(
    type='DefaultClassifier',
    num_classes=3,
    backbone_embed_dim=256,
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 4),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        # decoder unused with enc_mode=True, kept for compatibility
        dec_depths=(2, 2, 2),
        dec_channels=(64, 64, 128),
        dec_num_head=(4, 4, 8),
        dec_patch_size=(1024, 1024, 1024),
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
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            ignore_index=-1,
            # sqrt-inverse-frequency weights (pool proportions), sum ~ 3
            # order: Betula, Picea, Pinus  (alphabetical = class index order)
            weight=[1.42, 1.04, 0.54],
        ),
    ])

optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.02)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)

dataset_type = 'StandardizedDataset'
data_root = '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_3class_mixed_npy'
cache_data = False
class_names = ['Betula', 'Picea', 'Pinus']

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type='StandardizedDataset',
        split='train',
        data_root=data_root,
        class_names=class_names,
        label_level='genus',
        transform=[
            dict(type='CenterShiftMean'),
            # --- geometric augmentations ---
            dict(type='RandomRotate', angle=[-1, 1], center=None, axis='z',
                 always_apply=True, p=1.0),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1], anisotropic=True),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ElasticDistortion',
                 distortion_params=[[0.2, 0.3], [0.8, 1.0]]),
            # --- recompute grid_coord after augmentations ---
            dict(type='GridSample',
                 grid_size=0.02,
                 hash_type='fnv',
                 mode='train',
                 return_grid_coord=True),
            dict(type='ShufflePoint'),
            dict(type='ToTensor'),
            dict(type='Collect',
                 keys=('coord', 'grid_coord', 'category'),
                 feat_keys=['coord']),
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='StandardizedDataset',
        split='val',
        data_root=data_root,
        class_names=class_names,
        label_level='genus',
        transform=[
            dict(type='CenterShiftMean'),
            dict(type='GridSample',
                 grid_size=0.02,
                 hash_type='fnv',
                 mode='train',
                 return_grid_coord=True),
            dict(type='ToTensor'),
            dict(type='Collect',
                 keys=('coord', 'grid_coord', 'category'),
                 feat_keys=['coord']),
        ],
        test_mode=False))
