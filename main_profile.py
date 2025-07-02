import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('/root/RTDETR/ultralytics/cfg/models/rt-detr/rtdetr-slimneck-ASF.yaml')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[1280, 1280])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()