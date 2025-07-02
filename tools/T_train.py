from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    # 加载模型（RTDETR需要指定预训练权重路径）
    model = RTDETR(r'/root/ultralytics/X_datasets/rtdetr-resnet50.yaml')
    
    #.load('rtdetr-resnet50.pt')  # 推荐使用官方预训练权重

    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=r'/root/ultralytics/X_datasets/data.yaml',
        epochs=200,
        patience=50,
        batch=16,
        imgsz=1280,
        save=True,
        save_period=-1,
        cache=False,
        device='0',
        workers=4,
        project='X_results/train',
        name='UP1000_rtdetrrn50_0325',  # 修改实验名称以区分模型
        exist_ok=False,
        pretrained=True,  # 使用预训练权重（会自动加载模型对应的预训练权重）
        optimizer='AdamW',  # RTDETR推荐使用AdamW优化器
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,  # RTDETR建议最后10个周期关闭马赛克增强
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=10,  # 可以冻结backbone的前10层
        
        # 超参数 --------------------------------------------------------------------------------------------
        lr0=0.0001,  # RTDETR通常使用较小的学习率
        lrf=0.0001,
        weight_decay=0.0001,  # 通常比YOLO系列更小的权重衰减
        # momentum=0.9,  # AdamW优化器不使用momentum参数，但保留参数避免报错
        warmup_epochs=5.0,  # 适当延长预热时间
        
        # 数据增强参数（保留通用增强参数）---------------------------------------------------------------------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,  # 可以适当使用mixup增强
        
        # 移除不适用参数（RTDETR不需要以下参数）------------------------------------------------------------
        # box=7.5,        # RTDETR使用不同损失计算方式
        # cls=0.5,        # 分类损失权重自动处理
        # dfl=1.5,        # 无DFL损失
        # pose=12.0,      # 关键点检测相关
        # kobj=1.0,       # 关键点检测相关
        # overlap_mask=True,  # 分割相关参数
        # mask_ratio=4,   # 分割相关参数
        # nbs=64,         # 名义批量大小（自动处理）
        # warmup_momentum=0.8,  # AdamW不使用momentum
        # warmup_bias_lr=0.1    # 偏置预热（可选保留）
    )





