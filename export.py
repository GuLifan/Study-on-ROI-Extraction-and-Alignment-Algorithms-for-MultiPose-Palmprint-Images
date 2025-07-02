import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = RTDETR('/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune50%_vmup2400_250412-finetune/weights/best.pt')
    model.export(format='onnx', simplify=True)