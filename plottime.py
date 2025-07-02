import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 配置模型文件和标签
info_files = [
    '/root/RTDETR/runsyolo/val/exp_yolov8n_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runsyolo/val/exp_yolov8s_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runsyolo/val/exp_yolov8m_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runsyolo/val/exp_yolov10n_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runsyolo/val/exp_yolov10s_vmup2400_250411/paper_data.txt',

    '/root/RTDETR/runsyolo/val/exp_yolo11n_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runsyolo/val/exp_yolo11s_vmup2400_250411/paper_data.txt',

    '/root/RTDETR/runs1/val/exp_rtdetr-l_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runs1/val/exp_rtdetr-r18_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runs1/val/exp_rtdetr-r34_vmup2400_250411/paper_data.txt',
    '/root/RTDETR/runs1/val/exp_rtdetr-r50_vmup2400_250411/paper_data.txt',

]

custom_labels = [
    'YOLOv8n',
    'YOLOv8s',
    'YOLOv8m',
    'YOLOv10n',
    'YOLOv10s',
    'YOLO11n',
    'YOLO11s',
    'RT-DETR-l',
    'RT-DETR-R18',
    'RT-DETR-R34',
    'RT-DETR-R50',

]

def parse_info_file(file_path):
    """解析文本文件提取效率指标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到包含数值的数据行
        data_line = None
        for line in lines:
            if '|' in line and any(c.isdigit() for c in line):
                data_line = line.strip()
                break
        
        if not data_line:
            print(f"未找到数据行: {file_path}")
            return None
        
        # 分割并清理数据
        parts = [p.strip() for p in data_line.split('|') if p.strip()]
        
        # 调试输出
        print(f"解析文件: {file_path}")
        print(f"原始数据: {parts}")
        
        return {
            'GFLOPs': float(parts[0]),
            'Parameters': int(parts[1].replace(',', '')),
            'Preprocess': float(parts[2].replace('s', '')),
            'Inference': float(parts[3].replace('s', '')),
            'Postprocess': float(parts[4].replace('s', '')),
            'FPS_total': float(parts[5]),
            'FPS_inference': float(parts[6]),
            'Model_Size': float(parts[7].replace('MB', ''))
        }
    except Exception as e:
        print(f"解析错误 {file_path}: {str(e)}")
        return None

def plot_grouped_metrics(df, metrics_group, title, save_path):
    """绘制分组指标的柱状图"""
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    index = np.arange(len(df))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_group)))
    
    # 计算合适的y轴上限
    max_values = []
    for metric, _, mtype in metrics_group:
        if mtype == 'int':
            max_values.append(df[metric].max() / 1000000)  # 参数转换为百万单位
        else:
            max_values.append(df[metric].max())
    y_max = max(max_values) * 1.25  # 留出25%空间
    
    # 绘制每组柱状图
    for i, (metric, label, mtype) in enumerate(metrics_group):
        if mtype == 'int':
            values = df[metric] / 1000000  # 参数转换为百万单位
            unit = 'M'
        else:
            values = df[metric]
            unit = ''
        
        bars = plt.bar(index + i*bar_width, values, 
                      width=bar_width,
                      color=colors[i],
                      label=label)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if mtype == 'int':
                text = f'{height:.2f}{unit}'
            elif mtype == 'time':
                text = f'{height:.4f}s'
            else:
                text = f'{height:.2f}{unit}'
                
            plt.text(bar.get_x() + bar.get_width()/2,
                    height * 1.05,  # 略微抬高
                    text,
                    ha='center',
                    va='bottom',
                    rotation=75,    ##########################################数值旋转防止重叠
                    fontsize=8,
                    fontname='Times New Roman')  # 设置数值标签字体
    
    # 图表美化
    plt.title(title, pad=20, fontsize=14, fontname='Times New Roman')
    plt.xlabel('Models', labelpad=10, fontname='Times New Roman')
    
    # 根据指标类型设置y轴标签
    if any(m[2] == 'int' for m in metrics_group):
        plt.ylabel('Parameters (Millions)', labelpad=10, fontname='Times New Roman')
    else:
        plt.ylabel('Value', labelpad=10, fontname='Times New Roman')
        
    plt.xticks(index + bar_width*(len(metrics_group)-1)/2, 
               df['Model'], rotation=45, ha='right', fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')  # 设置y轴刻度字体
    plt.ylim(0, y_max)  # 动态设置y轴上限
    plt.grid(axis='y', alpha=0.3)
    
    # 设置图例字体
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

def plot_all_metrics(info_files, custom_labels, save_dir):
    """绘制并保存所有指标图"""
    # 收集数据
    all_metrics = []
    for file_path, label in zip(info_files, custom_labels):
        metrics = parse_info_file(file_path)
        if metrics:
            metrics['Model'] = label
            all_metrics.append(metrics)
            print(f"已加载: {label} - Parameters: {metrics['Parameters']}")
    
    if not all_metrics:
        print("没有有效数据可绘制！")
        return
    
    df = pd.DataFrame(all_metrics)
    os.makedirs(save_dir, exist_ok=True)
    
    # 打印数据框内容进行检查
    print("\n数据汇总:")
    print(df[['Model', 'Parameters']])
    
    # 定义分组指标
    time_metrics = [
        ('Preprocess', 'Preprocess (s)', 'time'),
        ('Inference', 'Inference (s)', 'time'),
        ('Postprocess', 'Postprocess (s)', 'time')
    ]
    
    fps_metrics = [
        ('FPS_total', 'FPS (total)', 'fps'),
        ('FPS_inference', 'FPS (inference)', 'fps')
    ]
    
    single_metrics = [
        ('GFLOPs', 'GFLOPs', 'float'),
        ('Parameters', 'Parameters (M)', 'int'),
        ('Model_Size', 'Model Size (MB)', 'size')
    ]
    
    # 绘制分组图表
    plot_grouped_metrics(df, time_metrics, 
                        'Inference Time Components Comparison',
                        os.path.join(save_dir, 'time_metrics.png'))
    
    plot_grouped_metrics(df, fps_metrics,
                        'FPS Performance Comparison',
                        os.path.join(save_dir, 'fps_metrics.png'))
    
    # 绘制单独指标
    for metric, label, mtype in single_metrics:
        plot_grouped_metrics(df, [(metric, label, mtype)],
                            f'{label} Comparison',
                            os.path.join(save_dir, f'{metric}.png'))
    
    # 保存原始数据
    data_path = os.path.join(save_dir, 'metrics_data.csv')
    df.to_csv(data_path, index=False)
    print(f"\n原始数据已保存: {data_path}")

if __name__ == '__main__':
    SAVE_DIR = "/root/RTDETR/plots/250415yushiyan"
    print("开始生成图表...")
    plot_all_metrics(info_files, custom_labels, SAVE_DIR)
    print("\n所有图表生成完成！")