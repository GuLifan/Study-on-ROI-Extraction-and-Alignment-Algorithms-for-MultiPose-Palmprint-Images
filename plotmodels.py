import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# 配置模型文件和标签
results_files = [
    
    '/root/RTDETR/runsyolo/train/exp_yolo11s_vmup2400_250411/results.csv',
   '/root/RTDETR/runsyolo/train/exp_yolo11n_vmup2400_250411/results.csv',
   '/root/RTDETR/runsyolo/train/exp_yolov10s_vmup2400_250411/results.csv',
 '/root/RTDETR/runsyolo/train/exp_yolov10n_vmup2400_250411/results.csv',
   '/root/RTDETR/runsyolo/train/exp_yolov8m_vmup2400_250411/results.csv',
 '/root/RTDETR/runsyolo/train/exp_yolov8s_vmup2400_250411/results.csv',
   '/root/RTDETR/runsyolo/train/exp_yolov8n_vmup2400_250411/results.csv',
   '/root/RTDETR/runs/train/exp_rtdetr-l_vmup2400_250411/results.csv',
   '/root/RTDETR/runs/train/exp_rtdetr-r50_vmup2400_250411/results.csv',
   '/root/RTDETR/runs/train/exp_rtdetr-r34_vmup2400_250411/results.csv',
   '/root/RTDETR/runs/train/exp_rtdetr-r18_vmup2400_250411/results.csv',
    '/root/RTDETR/runs/train/exp_rtdetr-r18-attention_vmup2400_250411/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune05%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune10%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune20%_vmup2400_250412-finetune/results.csv',
  '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune30%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune40%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune50%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune60%_vmup2400_250412-finetune/results.csv',
    '/root/RTDETR/runs/prune/exp_rtdetr-r18-attention-prune80%_vmup2400_250412-finetune/results.csv',
    
    
]

custom_labels = [
   'yolo11s',
   'yolo11n',
   'yolov10s',
   'yolov10n',
   'yolov8m',
   'yolov8s',
   'yolov8n',
  'rtdetr-l',
   'rtdetr-r50',
   'rtdetr-r34',
   'rtdetr-r18',
'rtdetr-r18-attention',
    'rtdetr-r18-attention-5%',
    'rtdetr-r18-attention-10%',
    'rtdetr-r18-attention-20%',
   'rtdetr-r18-attention-30%',
    'rtdetr-r18-attention-40%',
    'rtdetr-r18-attention-50%',
    'rtdetr-r18-attention-60%',
   'rtdetr-r18-attention-80%',
    
]

def calculate_f1(precision, recall):
    """计算F1分数"""
    return 2 * (precision * recall) / (precision + recall + 1e-16)

def plot_metrics_bar_chart(results_files, custom_labels, save_dir=None):
    """
    绘制模型指标对比柱状图（优化标签防重叠版）
    
    参数:
        results_files (list): 结果文件路径列表
        custom_labels (list): 模型标签列表
        save_dir (str): 图片保存目录
    """
    # 准备存储各模型指标
    metrics_data = {
        'Model': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'mAP50': [],
        'mAP50-95': []
    }
    
    # 提取每个模型的最终指标
    for file_path, label in zip(results_files, custom_labels):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # 清理列名空格
            
            # 获取最终epoch的数据（最后一行）
            final_row = df.iloc[-1]
            
            # 计算F1分数
            precision = final_row['metrics/precision(B)']
            recall = final_row['metrics/recall(B)']
            f1 = calculate_f1(precision, recall)
            
            # 存储数据
            metrics_data['Model'].append(label)
            metrics_data['F1'].append(f1)
            metrics_data['Precision'].append(precision)
            metrics_data['Recall'].append(recall)
            metrics_data['mAP50'].append(final_row['metrics/mAP50(B)'])
            metrics_data['mAP50-95'].append(final_row['metrics/mAP50-95(B)'])
            
        except Exception as e:
            print(f"Error processing {label}: {str(e)}")
            continue
    
    # 转换为DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    
    # 设置绘图参数
    plt.figure(figsize=(16, 8))
    bar_width = 0.15
    index = np.arange(len(custom_labels))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制柱状图
    metrics_to_plot = ['F1', 'Precision', 'Recall', 'mAP50', 'mAP50-95']
    for i, metric in enumerate(metrics_to_plot):
        bars = plt.bar(index + i*bar_width, df_metrics[metric], 
                      width=bar_width, 
                      color=colors[i],
                      label=metric)
    
        # 添加旋转的数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                     height + 0.02, 
                     f'{height:.3f}', 
                     ha='center', 
                     va='bottom',
                     rotation=60,  # 旋转45度
                     fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))
    
    # 图表美化
    plt.title('Model Performance Comparison', pad=20, fontsize=14)
    plt.xlabel('Models', labelpad=10)
    plt.ylabel('Score', labelpad=10)
    plt.xticks(index + bar_width*2, custom_labels, rotation=15)
    plt.ylim(0, 1.15)  # 增加顶部空间
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # 调整边距防止标签被裁剪
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 右侧留出图例空间
    
    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'model_comparison_rotated_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return df_metrics

if __name__ == '__main__':
    # 设置保存目录
    SAVE_DIR = "/root/RTDETR/plots/full"
    
    # 绘制柱状图并获取数据
    metrics_df = plot_metrics_bar_chart(
        results_files=results_files,
        custom_labels=custom_labels,
        save_dir=SAVE_DIR
    )
    
    # 打印数值表格
    print("\nPerformance Metrics Table:")
    print(metrics_df.to_markdown(index=False, floatfmt=".3f"))