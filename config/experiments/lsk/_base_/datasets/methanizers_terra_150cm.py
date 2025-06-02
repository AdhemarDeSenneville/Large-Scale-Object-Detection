# dataset settings
dataset_type = 'DOTADataset'
data_root = "/home/adhemar/Bureau/datasets/Methanizers/Terra_Data_1/" 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ('all', 'tank', 'pile')


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        filter_empty_gt = False,
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'train/labels/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        filter_empty_gt = False,
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        filter_empty_gt = False,
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'val/labels/', # 'test_split/labels/'
        img_prefix=data_root + 'val/images/', #  'test_split/images/'
        pipeline=test_pipeline))
