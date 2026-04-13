"""
PTv3 + AlphaEarth + SINR deterministic concat fusion — 10-class dual-val, 4-GPU / 120-epoch.
  Late concat fusion: PTv3 (512) + AEEncoder (256) + SINREncoder (256) → 1024 → num_classes
  Architecture: MultiCatCtxCls-v1m1 — both sources always active, no dropout, no auxiliary heads.
  Dataset, loss, backbone, optimizer identical to cls-ptv3-ctx-ae-sinr-cat-10class-dual-val-4gpu.py.
  CtxVMFAugment (kappa=100) applied to ctx_ae during training; ctx_sinr unperturbed.
"""

weight = None
resume = False
evaluate = True
test_only = False
seed = 3313067
save_path = '/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/Pointcept/exp/snapshot_10class_dual_val/ptv3_ctx_ae_sinr_cat_4gpu_120ep'
num_worker = 16
batch_size = 128
gradient_accumulation_steps = 1
batch_size_val = 256
batch_size_test = None
epoch = 120
eval_epoch = 120
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
param_dicts = [dict(keyword='block', lr=0.0004)]

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='DualValClsEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
]

train = dict(type='DefaultTrainer')

model = dict(
    type='MultiCatCtxCls-v1m1',
    num_classes=10,
    backbone_embed_dim=512,
    context_embed_dim=256,
    context_keys=['ctx_ae', 'ctx_sinr'],
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
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
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    context_encoders=[
        dict(type='AEEncoder', input_dim=64, output_dim=256),
        dict(type='SINREncoder', input_dim=256, output_dim=256),
    ],
    criteria=[
        dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            ignore_index=-1,
            # sqrt-inverse-frequency weights (pool proportions), sum ~ 10
            # order: Abies, Acer, Alnus, Betula, Carpinus, Fagus, Larix, Picea, Pinus, Quercus
            weight=[0.74, 1.74, 1.60, 0.59, 1.40, 0.70, 1.79, 0.43, 0.22, 0.78],
        ),
    ])

optimizer = dict(type='AdamW', lr=0.004, weight_decay=0.02)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.004, 0.0004],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)

dataset_type = 'StandardizedDataset'
data_root = '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_10class_dual_val_npy'
context_pth = '/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/data/snapshot_v1/context_features.pth'
cache_data = False
class_names = ['Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
               'Fagus', 'Larix', 'Picea', 'Pinus', 'Quercus']

data = dict(
    num_classes=10,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type='StandardizedDataset',
        split='train',
        data_root=data_root,
        class_names=class_names,
        label_level='genus',
        context_pth=context_pth,
        context_sources=['alphaearth', 'sinr'],
        transform=[
            dict(type='CenterShiftMean'),
            dict(type='RandomRotate', angle=[-1, 1], center=None, axis='z',
                 always_apply=True, p=1.0),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1], anisotropic=True),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ElasticDistortion',
                 distortion_params=[[0.2, 0.3], [0.8, 1.0]]),
            dict(type='GridSample',
                 grid_size=0.02,
                 hash_type='fnv',
                 mode='train',
                 return_grid_coord=True),
            dict(type='ShufflePoint'),
            dict(type='ToTensor'),
            dict(type='CtxVMFAugment', key='ctx_ae', kappa=100.0),
            dict(type='Collect',
                 keys=('coord', 'grid_coord', 'category', 'ctx_ae', 'ctx_sinr'),
                 feat_keys=['coord']),
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='StandardizedDataset',
        split='val_id',
        data_root=data_root,
        class_names=class_names,
        label_level='genus',
        context_pth=context_pth,
        context_sources=['alphaearth', 'sinr'],
        transform=[
            dict(type='CenterShiftMean'),
            dict(type='GridSample',
                 grid_size=0.02,
                 hash_type='fnv',
                 mode='train',
                 return_grid_coord=True),
            dict(type='ToTensor'),
            dict(type='Collect',
                 keys=('coord', 'grid_coord', 'category', 'ctx_ae', 'ctx_sinr', 'source_id'),
                 feat_keys=['coord']),
        ],
        test_mode=False))
