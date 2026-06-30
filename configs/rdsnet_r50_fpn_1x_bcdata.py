norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)

model = dict(
    type="RDSNet",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch"),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type="RdsRetinaHead",
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0)),
    mask_head=dict(
        type="RdsMaskHead",
        num_ins=5,
        end_level=2,
        in_channels=256,
        conv_out_channels=256,
        num_convs=1,
        norm_cfg=norm_cfg))

train_cfg = dict(
    assigner=dict(
        type="MaxIoUAssigner",
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    masks_to_train=100,
    crop=dict(padding=0.25),
    test_crop=dict(padding=0.1),
    mask_loss=dict(use_mask_ohem=True, pos_neg_ratio=1, loss_weight=1),
    upsampling=False,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type="nms", iou_thr=0.5),
    max_per_img=100,
    crop=dict(padding=0.1),
    mask_thr_binary=0.4)

dataset_type = "CocoDataset"
data_root = "/data/zju-151/liyixin/project/pathology-cell-counting/datasets/processed/bcdata/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ])
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        img_prefix=data_root + "images/",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "images/",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "images/",
        pipeline=test_pipeline))

optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook")])

total_epochs = 20
device_ids = range(1)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "/data/zju-151/liyixin/project/pathology-cell-counting/results/work_dirs/densenucleidet_bcdata"
load_from = None
resume_from = None
workflow = [("train", 1)]
