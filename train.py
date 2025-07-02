import os
import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/root/RTDETR/ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-DAF.yaml')
    # model.load('/root/RTDETR/weights/rtdetr-l.pt') # loading pretrain weights
    model.train(data='/root/RTDETR/datasets/data.yaml',
                cache=False,
                imgsz=1080,
                epochs=300,
                patience=50,
                batch=16, # 高于4的时候会掉性能（全都按8以缩短时间）
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0，默认是4
                device='0',
                amp=True,# 这个版本的最好不使用AMP，打开以缩短训练时间，增大batchsize
                warmup_epochs=10,  # 热身epoch数量，我设为10，一般来说YOLO家族的模型都是3（较长，可能是 DETR 特性）
                # resume='', # last.pt path
                project='runs/train',
                name='exp_rtdetr-r18-MutilBackbone-DAF_vmup2400_250414',
                )
    

# 欸呀吗，这不attention吗？
# 还是看看远处的多头注意力机制教程吧家人们！
# ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-DAF.yaml