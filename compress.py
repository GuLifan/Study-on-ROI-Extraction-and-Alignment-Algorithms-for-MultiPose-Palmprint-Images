import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.compress import RTDETRCompressor, RTDETRFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 50
    compressor = RTDETRCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = RTDETRFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/root/RTDETR/runs/train/exp_rtdetr-r18-MutilBackbone-DAF_vmup2400_250414/weights/best.pt',
        'data':'/root/RTDETR/datasets/data.yaml',
        'imgsz': 1080,
        'epochs': 300,
        'batch': 8,
        'workers': 8,
        'cache': False,
        'device': '0',
        'project':'runs/prune',
        'name':'exp_rtdetr-r18-MutilBackbone-DAF-prune80%_vmup2400_250414',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True, # 全局剪枝参数
        'speed_up': 1.25, # 剪枝前计算量/剪枝后计算量
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
        'iterative_steps': 50
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)