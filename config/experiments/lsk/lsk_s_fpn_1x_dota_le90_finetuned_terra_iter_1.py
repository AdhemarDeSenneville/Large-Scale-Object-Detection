_base_ = [
    './lsk_s_fpn_1x_dota_le90.py',
    './_base_/datasets/methanizers_terra_150cm_iter_1.py', # WARNING Changed
    './_base_/schedules/schedule_3x.py', # WARNING Changed
    './_base_/default_runtime.py'
]

angle_version = 'le90'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
)

# --------------- #
#      DATA       # 
# --------------- #

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CenterCrop', crop_size=(500, 500), crop_type='absolute'),
    #dict(type='RandomCrop', crop_size=(500, 500), crop_type='absolute'),
    dict(type='RResize', img_scale=(1024, 1024)),

    #dict(
    #    type='PhotoMetricDistortion',
    #    brightness_delta=32,            # max brightness shift
    #    contrast_range=(0.5, 1.5),      # contrast multiplier range
    #    saturation_range=(0.5, 1.5),    # saturation multiplier range
    #    hue_delta=18                    # max hue shift
    #),

    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version
    ),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomCrop', crop_size=(500, 500), crop_type='absolute'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            #dict(type='CenterCrop', crop_size=(500, 500), crop_type='absolute'),
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


data = dict(
    train=dict(
        pipeline=train_pipeline, 
        version=angle_version,
        ),
    val=dict(
        pipeline=test_pipeline,
        version=angle_version,
        ),
    test=dict(
        pipeline=test_pipeline,
        version=angle_version,
        )
)



default_hooks = dict(logger=dict(interval=100)),
load_from = '/home/adhemar/Bureau/METHAN/model/DOTA_pretrained/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth'  # noqa