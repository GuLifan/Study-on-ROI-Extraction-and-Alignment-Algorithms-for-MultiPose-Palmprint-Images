import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
# model_scale = 'l'  # 改为需要的规模，v8v11需要指定，其他的一般不用

if __name__ == '__main__':
    model = RTDETR(r'/root/ultralytics/ultralytics/rtdetr-r50.yaml')
    #model = RTDETR(r'/root/ultralytics/rtdetr-r50.pt')  # 或 rtdetr-x.pt, rtdetr-r50.pt
    # model.load('') # loading pretrain weights
    model.train(data=r'/root/ultralytics/datasets/data.yaml',
                cache=False,
                imgsz=1280,
                epochs=300,  # 覆盖了default中的100
                batch=8,
                workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0', # 代表着这是第一块GPU，实际上我就一块GPU
                # resume='', # last.pt path
                project='runs/train',
                name='exp_rtdetr-r50_up1000_100_250331', #命名规则是exp_加载的模型yaml文件_数据集_epoch数量_batchsize_日期六位
                amp=True, # 我这里还是选择打开amp以提高训练效率，在有限循环内实现
                task='detect',
                mode='train',
                patience=50,  # 早期停止等待epoch数
                save=True,
                save_period=-1,
                exist_ok=False,
                pretrained=True,
                optimizer='AdamW',
                verbose=True,
                seed=0,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=False,
                close_mosaic=0,
                resume=False,
                fraction=1.0,
                profile=False,
                freeze=None,
                
                # 验证参数
                val=True,
                split='val',
                save_json=False,
                save_hybrid=False,
                iou=0.7,
                max_det=300,
                half=False,
                dnn=False,
                plots=True,
                
                # 超参数
                lr0=0.0001,  # 初始学习率
                lrf=1.0,  # 最终学习率(lr0 * lrf)
                momentum=0.9,
                weight_decay=0.0001,
                warmup_epochs=10,  # 热身迭代次数我调整为了10，避免梯度剧烈震荡
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,  # box损失增益
                cls=0.5,  # cls损失增益
                dfl=1.5,  # dfl损失增益
                label_smoothing=0.0,
                nbs=64,  # 标称批量大小
                
                # 数据增强参数
                hsv_h=0.015,  # 图像HSV-Hue增强
                hsv_s=0.7,  # 图像HSV-Saturation增强
                hsv_v=0.4,  # 图像HSV-Value增强
                degrees=0.0,  # 图像旋转(+/- deg)
                translate=0.1,  # 图像平移(+/- fraction)
                scale=0.5,  # 图像缩放(+/- gain)
                shear=0.0,  # 图像剪切(+/- deg)
                perspective=0.0,  # 图像透视(+/- fraction)
                flipud=0.0,  # 图像上下翻转概率
                fliplr=0.5,  # 图像左右翻转概率
                mosaic=0.0,  # 图像马赛克概率
                mixup=0.0,  # 图像mixup概率
                copy_paste=0.0,  # 分割复制粘贴概率
                )