
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# 训练结果文件列表
results_files = [
    '/root/RTDETR/runsyolo/train/exp_yolo11s_vmup2400_250411/results.csv',
    '/root/RTDETR/runs/train/exp_rtdetr-r18_vmup2400_250411/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune05%_vmup2400_250412-finetune/results.csv',

    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune40%_vmup2400_250412-finetune/results.csv',

    '/root/RTDETR/runs/train/exp_rtdetr-r18-attention_vmup2400_250411/results.csv',
]

# 自定义图例标签（与results_files顺序对应）
custom_labels = [
    'yolo11s',
        'rtdetr-r18',
    'rtdetr-r18-attention-05%',


    'rtdetr-r18-attention-40%',

    'rtdetr-r18-attention',
]

def plot_metric_comparison(metric_key, metric_label, custom_labels):
    """保持原始单图单指标风格，适配RT-DETR列名，自动保存图片"""
    plt.figure(figsize=(10, 6))
    
    # 列名映射规则（YOLO列名 -> RT-DETR实际列名）
    column_mapping = {
        'train/box_loss': 'train/giou_loss',
        'val/box_loss': 'val/giou_loss',
        'train/dfl_loss': 'train/l1_loss',
        'val/dfl_loss': 'val/l1_loss'
    }
    target_column = column_mapping.get(metric_key, metric_key)

    for file_path, custom_label in zip(results_files, custom_labels):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            if 'epoch' not in df.columns:
                print(f"Missing 'epoch' in {os.path.basename(file_path)}")
                continue

            if target_column not in df.columns:
                print(f"Missing '{target_column}' in {os.path.basename(file_path)}")
                continue

            plt.plot(df['epoch'], df[target_column], label=custom_label)

        except Exception as e:
            print(f"Error in {os.path.basename(file_path)}: {str(e)}")
            continue

    plt.title(f'{metric_label} Comparison')
    plt.xlabel('Epochs')
    plt.ylabel(metric_label)
    plt.legend()
    
    # 自动保存到指定目录
    save_dir = "/root/RTDETR/plots/onebyone"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{metric_key.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()

if __name__ == '__main__':
    # 精度指标（保持YOLO风格列名，实际自动映射）
    metrics = [
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'mAP@50'),
        ('metrics/mAP50-95(B)', 'mAP@50-95')
    ]

    # 损失函数（使用YOLO风格列名，自动映射到RT-DETR列）
    loss_metrics = [
        ('train/box_loss', 'Train GIoU Loss'),
        ('train/cls_loss', 'Train Class Loss'),
        ('train/dfl_loss', 'Train L1 Loss'),
        ('val/box_loss', 'Val GIoU Loss'),
        ('val/cls_loss', 'Val Class Loss'),
        ('val/dfl_loss', 'Val L1 Loss')
    ]

    # 创建输出目录
    os.makedirs("/root/RTDETR/plots/onebyone", exist_ok=True)

    print("===== 精度指标对比 =====")
    for metric, label in metrics:
        plot_metric_comparison(metric, label, custom_labels)

    print("\n===== 损失函数对比 =====")
    for metric, label in loss_metrics:
        plot_metric_comparison(metric, label, custom_labels)

    print("\n所有图片已保存至 /root/RTDETR/plots/onebyone")