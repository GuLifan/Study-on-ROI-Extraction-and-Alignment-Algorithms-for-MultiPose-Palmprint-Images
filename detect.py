import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = RTDETR('/root/RTDETR/runs/prune/exp_rtdetr-r18-MutilBackbone-DAF-prune80%_vmup2400_250414-finetune/weights/best.pt') # select your model.pt path
    model.predict(source='/root/RTDETR/images',
                  conf=0.25,
                  project='runs/detect',
                  name='exp_rtdetr-r18-MutilBackbone-DAF-prune80%_vmup2400_250414-finetune',
                  save=True,
                  # visualize=True # visualize model features maps/root/RTDETR/runs/prune/exp_rtdetr-r18_attention_prunex1.5_mj_up1000_250405-finetune/weights/best.pt
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )