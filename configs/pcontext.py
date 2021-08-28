# dataset settings
dataset_type = 'PascalContextDataset'
data_root = '/home/ubuntu/dataset/PASCAL_Context/processed/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (520, 520)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'distance_map', 'angle_map']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/image',
        ann_dir='train/label',
        dt_dir='train/offset_gt/dt_offset',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/label',
        dt_dir='val/offset_gt/dt_offset',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/label',
        dt_dir='val/offset_gt/dt_offset',
        pipeline=test_pipeline))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='ms_deit_segfix',
    feat_dim=256,
    k=16,
    L=3,
    dropout=0.1,
    heads=8,
    hidden_dim=2048,
    depth=2,
    loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='slide', num_classes=60, stride=(160,160), crop_size=(480, 480), num_queries=3600)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8,
                 paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=126000, by_epoch=False)
# runtime settings
# total_iters = 640000
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU')


# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True