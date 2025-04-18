img_size=(992, 736)
tile_cfg=dict(
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.5,
    max_per_img=1500,
    filter_empty_gt=True)
img_norm_cfg=dict(
    mean=[0, 0, 0],
    std=[255, 255, 255],
    to_rgb=True)
train_pipeline=[
    dict(type='Resize',
        img_scale=(992, 736),
        keep_ratio=False),
    dict(type='Normalize',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        to_rgb=True),
    dict(type='RandomFlip',
        flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'])]
test_pipeline=[
    dict(type='MultiScaleFlipAug',
        img_scale=(992, 736),
        flip=False,
        transforms=[
            dict(type='Resize',
                keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='ImageToTensor',
                keys=['img']),
            dict(type='Collect',
                keys=['img'])])]
train_dataset=dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='OTXDetDataset',
        pipeline=[
            dict(type='LoadImageFromOTXDataset',
                enable_memcache=True),
            dict(type='LoadAnnotationFromOTXDataset',
                with_bbox=True)]),
    pipeline=[
        dict(type='Resize',
            img_scale=(992, 736),
            keep_ratio=False),
        dict(type='Normalize',
            mean=[0, 0, 0],
            std=[255, 255, 255],
            to_rgb=True),
        dict(type='RandomFlip',
            flip_ratio=0.5),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect',
            keys=['img', 'gt_bboxes', 'gt_labels'],
            meta_keys=['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'])],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.5,
    max_per_img=1500,
    filter_empty_gt=True)
val_dataset=dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='OTXDetDataset',
        pipeline=[
            dict(type='LoadImageFromOTXDataset',
                enable_memcache=True),
            dict(type='LoadAnnotationFromOTXDataset',
                with_bbox=True)]),
    pipeline=[
        dict(type='MultiScaleFlipAug',
            img_scale=(992, 736),
            flip=False,
            transforms=[
                dict(type='Resize',
                    keep_ratio=False),
                dict(type='RandomFlip'),
                dict(type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='ImageToTensor',
                    keys=['img']),
                dict(type='Collect',
                    keys=['img'])])],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.5,
    max_per_img=1500,
    filter_empty_gt=True)
test_dataset=dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='OTXDetDataset',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromOTXDataset')]),
    pipeline=[
        dict(type='MultiScaleFlipAug',
            img_scale=(992, 736),
            flip=False,
            transforms=[
                dict(type='Resize',
                    keep_ratio=False),
                dict(type='RandomFlip'),
                dict(type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='ImageToTensor',
                    keys=['img']),
                dict(type='Collect',
                    keys=['img'])])],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.5,
    max_per_img=1500,
    filter_empty_gt=True)
data=dict(
    train=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='OTXDetDataset',
            pipeline=[
                dict(type='LoadImageFromOTXDataset',
                    enable_memcache=True),
                dict(type='LoadAnnotationFromOTXDataset',
                    with_bbox=True)]),
        pipeline=[
            dict(type='Resize',
                img_scale=(992, 736),
                keep_ratio=False),
            dict(type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='RandomFlip',
                flip_ratio=0.5),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'])],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.5,
        max_per_img=1500,
        filter_empty_gt=True),
    val=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='OTXDetDataset',
            pipeline=[
                dict(type='LoadImageFromOTXDataset',
                    enable_memcache=True),
                dict(type='LoadAnnotationFromOTXDataset',
                    with_bbox=True)]),
        pipeline=[
            dict(type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize',
                        keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='ImageToTensor',
                        keys=['img']),
                    dict(type='Collect',
                        keys=['img'])])],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.5,
        max_per_img=1500,
        filter_empty_gt=True),
    test=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='OTXDetDataset',
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromOTXDataset')]),
        pipeline=[
            dict(type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize',
                        keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='ImageToTensor',
                        keys=['img']),
                    dict(type='Collect',
                        keys=['img'])])],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.5,
        max_per_img=1500,
        filter_empty_gt=True))