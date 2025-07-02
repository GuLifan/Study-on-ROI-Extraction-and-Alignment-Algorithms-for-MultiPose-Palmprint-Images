import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# 训练结果文件列表
results_files = [
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune05%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune10%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune20%_vmup2400_250412-finetune/results.csv',
        '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune30%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune40%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune50%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune60%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/prune/exp_rtdetr-r18-attention-prune80%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs1/train/exp_rtdetr-r18-attention_vmup2400_250411/results.csv',
]

# 自定义图例标签（与results_files顺序对应）
custom_labels = [
    'RT-DETR-R18 05%',
        'RT-DETR-R18 10%',
            'RT-DETR-R18 20%',
                'RT-DETR-R18 30%',
                    'RT-DETR-R18 40%',
                        'RT-DETR-R18 50%',
                            'RT-DETR-R18 60%',
                            'RT-DETR-R18 80%',
                            'RT-DETR-R18',
                            

]

def plot_comparison(metrics, labels, custom_labels, layout=(2, 2), save_dir=None):
    """
    绘制指标对比图并保存
    
    参数:
        metrics (list): 要绘制的指标列名（如 'metrics/precision(B)'）
        labels (list): 指标对应的显示标签（如 'Precision'）
        custom_labels (list): 每条曲线的自定义标签（如模型名称）
        layout (tuple): 子图布局 (rows, cols)
        save_dir (str): 图片保存目录（如果为None则不保存）
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        for file_path, custom_label in zip(results_files, custom_labels):
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # 清理列名空格

                if 'epoch' not in df.columns:
                    print(f"Warning: 'epoch' column missing in {file_path}")
                    continue

                if metric_key not in df.columns:
                    print(f"Warning: '{metric_key}' column missing in {file_path}")
                    continue

                axes[i].plot(df['epoch'], df[metric_key], label=custom_label)

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        axes[i].set_title(metric_label)
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric_label)
        axes[i].legend()

    plt.tight_layout()
    
    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 替换文件名中的特殊字符
        safe_metrics = [m.replace('/', '_').replace('(', '').replace(')', '') for m in metrics[:2]]
        filename = f"plot_{'_'.join(safe_metrics)}_{timestamp}.png"  # 示例: plot_metrics_precision_B_metrics_recall_B_20240413_142022.png
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # 设置图片保存目录
    SAVE_DIR = "/root/RTDETR/plots/r18p"  # 修改为您需要的路径
    
    # 1. 绘制精度指标对比图
    plot_comparison(
        metrics=['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
        labels=['Precision', 'Recall', 'mAP@50', 'mAP@50-95'],
        custom_labels=custom_labels,
        layout=(2, 2),
        save_dir=SAVE_DIR
    )
    
    # 2. 绘制损失函数对比图（适配RT-DETR实际列名）
    plot_comparison(
        metrics=['train/giou_loss', 'train/cls_loss', 'train/l1_loss', 'val/giou_loss', 'val/cls_loss', 'val/l1_loss'],
        labels=['Train GIoU Loss', 'Train Class Loss', 'Train L1 Loss', 'Val GIoU Loss', 'Val Class Loss', 'Val L1 Loss'],
        custom_labels=custom_labels,
        layout=(2, 3),
        save_dir=SAVE_DIR
    )