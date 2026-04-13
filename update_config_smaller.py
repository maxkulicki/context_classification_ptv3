weight = None
resume = False
evaluate = True
test_only = False
seed = 3313067
save_path = '/net/pr2/projects/plgrid/plggtreeseg/context_classification_ptv3/Pointcept/exp/snapshot1/ptv3_baseline_8gpu_100ep'
num_worker = 32
batch_size = 256
gradient_accumulation_steps = 1
batch_size_val = 512
batch_size_test = None
epoch = 100
eval_epoch = 100
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
param_dicts = [dict(keyword='block', lr=0.0001)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='ClsEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=True)
]
train = dict(type='DefaultTrainer')

#remove Voting tester
test = dict(type='ClsVotingTester', verbose=True, num_repeat=10)

model = dict(
    type='DefaultClassifier',
    num_classes=13,
    backbone_embed_dim=512,   #128
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        #enc_channels=(32, 64, 128, 256),
        #smaller: enc_channels=(32, 64, 64, 128)
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 2, 4, 8, 16),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        #dec_channels=(96, 96, 96, 96),
        #dec_channels=(96, 96, 96, 96),

        dec_channels=(64, 64, 128, 256),

        #dec_num_head=(4, 4, 4, 4),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0, #increase?
        proj_drop=0.0, #increase?
        drop_path=0.3, 
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,    #does it even use the decoder?
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    criteria=[
        #change to WCE, remove Lovasz
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss', 
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.008, weight_decay=0.01)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.008, 0.0008],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'StandardizedDataset'
data_root = '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_1_npy_fps8192'
cache_data = False
class_names = [
    'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus', 'Fagus', 'Fraxinus',
    'Larix', 'Picea', 'Pinus', 'Pseudotsuga', 'Quercus', 'Tilia'
]
data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus', 'Fagus', 'Fraxinus',
        'Larix', 'Picea', 'Pinus', 'Pseudotsuga', 'Quercus', 'Tilia'
    ],
    train=dict(
        type='StandardizedDataset',
        split='train',
        data_root=
        '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_1_npy_fps8192',
        class_names=[
            'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus', 'Fagus',
            'Fraxinus', 'Larix', 'Picea', 'Pinus', 'Pseudotsuga', 'Quercus',
            'Tilia'
        ],
        label_level='genus',
        transform=[
            dict(type='CenterShiftMean'),
            #dict(type='RandomScale', scale=[0.9, 1.1], anisotropic=True),
            dict(type='RandomScale', scale=[0.7, 1.5], anisotropic=True),
            dict(
                type='RandomShift',
                shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            #co to robi?
            dict(type='ShufflePoint'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'category'),
                feat_keys=['coord'])
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='StandardizedDataset',
        split='val',
        data_root=
        '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_1_npy_fps8192',
        class_names=[
            'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus', 'Fagus',
            'Fraxinus', 'Larix', 'Picea', 'Pinus', 'Pseudotsuga', 'Quercus',
            'Tilia'
        ],
        label_level='genus',
        transform=[
            dict(type='CenterShiftMean'),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'category'),
                feat_keys=['coord'])
        ],
        test_mode=False),
    test=dict(
        type='StandardizedDataset',
        split='val',
        data_root=
        '/net/pr2/projects/plgrid/plggtreeseg/data/snapshot_1_npy_fps8192',
        class_names=[
            'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus', 'Fagus',
            'Fraxinus', 'Larix', 'Picea', 'Pinus', 'Pseudotsuga', 'Quercus',
            'Tilia'
        ],
        label_level='genus',
        transform=[dict(type='CenterShiftMean')],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type='GridSample',
                    grid_size=0.02,
                    hash_type='fnv',
                    mode='train',
                    return_grid_coord=True),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord'),
                    feat_keys=['coord'])
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [1, 1],
                'anisotropic': True
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.8, 1.2],
                               'anisotropic': True
                           }]])))
