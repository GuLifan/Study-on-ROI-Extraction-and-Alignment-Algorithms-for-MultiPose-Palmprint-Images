import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ==================== 字体配置部分 ====================
def configure_font():
    """配置字体，优先使用Times New Roman，失败时使用系统默认衬线字体"""
    try:
        # 添加字体路径
        font_dir = '/root/RTDETR/fonts'
        font_files = fm.findSystemFonts(fontpaths=font_dir)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
        
        # 尝试直接设置Times New Roman
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = 'Times New Roman'
        rcParams['font.weight'] = 'normal'
        
        # 验证字体是否可用
        test_font = fm.FontProperties(family='serif', style='normal')
        if not any('Times' in f.name for f in fm.fontManager.ttflist if f.name == test_font.get_name()):
            raise ValueError("Times New Roman not found")
            
        print("字体设置成功: Times New Roman")
    except Exception as e:
        print(f"字体设置错误: {str(e)}")
        # 获取系统所有可用字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        # 寻找合适的替代字体
        fallback_fonts = [
            'DejaVu Serif', 
            'Liberation Serif',
            'Nimbus Roman',
            'Century Schoolbook L',
            'Georgia',
            'Palatino'
        ]
        
        selected_font = None
        for font in fallback_fonts:
            if font in available_fonts:
                selected_font = font
                break
                
        if selected_font is None:
            selected_font = 'serif'  # 最终回退到通用衬线字体
            
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = selected_font
        print(f"Times New Roman不可用，已替换为: {selected_font}")

# ==================== 数据解析部分 ====================
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
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune30%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune40%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune50%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune60%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune70%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF-prune80%_vmup2400_250414-finetune/paper_data.txt',
    '/root/RTDETR/runs/val/exp_rtdetr-r18-MutilBackbone-DAF_vmup2400_250414/paper_data.txt',
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
          'RT-DETR-R18-MutilBackbone-DAF 30%',
          'RT-DETR-R18-MutilBackbone-DAF 40%',
          'RT-DETR-R18-MutilBackbone-DAF 50%',
          'RT-DETR-R18-MutilBackbone-DAF 60%',
          'RT-DETR-R18-MutilBackbone-DAF 70%',
          'RT-DETR-R18-MutilBackbone-DAF 80%',
          'RT-DETR-R18-MutilBackbone-DAF',
]

def parse_metrics_file(file_path):
    """解析文本文件提取模型性能指标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割Model Info和Model Metrice部分
        parts = content.split('Model Metrice')
        if len(parts) < 2:
            return None
            
        metrics_section = parts[1]
        lines = [line.strip() for line in metrics_section.split('\n') if line.strip()]
        
        metrics_data = {}
        for line in lines:
            if line.startswith('+') or 'Class Name' in line:
                continue
            
            # 处理数据行
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 7:
                class_name = parts[0].strip().lower()
                # 特殊处理"all(平均数据)"行
                if '(平均数据)' in class_name or 'all' in class_name:
                    class_name = 'all'
                else:
                    class_name = class_name.split()[0]  # 只取第一部分作为类名
                
                try:
                    metrics_data[class_name] = {
                        'Precision': float(parts[1]),
                        'Recall': float(parts[2]),
                        'F1-Score': float(parts[3]),
                        'mAP50': float(parts[4]),
                        'mAP75': float(parts[5]),
                        'mAP50-95': float(parts[6])
                    }
                except ValueError as e:
                    print(f"数值转换错误 {file_path} 行: {line} - {str(e)}")
                    continue
        
        return metrics_data
    except Exception as e:
        print(f"解析错误 {file_path}: {str(e)}")
        return None

# ==================== 绘图部分 ====================
def plot_metrics_comparison(metrics_data, class_names, save_dir):
    """绘制每个类别的性能指标比较图"""
    # 配置字体
    configure_font()
    
    # 定义颜色方案
    colors = ['#205EA7', '#EAF0B2', '#38B6C4', '#965478', '#C8E2B2', '#1D91C2']

    metric_names = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    
    # 为每个类别创建单独的图表
    for class_name in class_names:
        plt.figure(figsize=(14, 7))
        plt.subplots_adjust(top=0.95)
        
        # 收集所有模型的数据
        models = []
        metric_values = {name: [] for name in metric_names}
        
        for model_label, model_metrics in metrics_data.items():
            if class_name in model_metrics:
                models.append(model_label)
                for name in metric_names:
                    metric_values[name].append(model_metrics[class_name][name])
        
        if not models:  # 如果没有该类的数据则跳过
            continue
        
        # 设置x轴位置
        x = np.arange(len(models))
        width = 0.12
        
        # 绘制每种指标
        for i, (name, color) in enumerate(zip(metric_names, colors)):
            plt.bar(x + width*(i-2.5), metric_values[name], width, 
                   color=color, label=name)
        
        # 添加数值标签
        for i, name in enumerate(metric_names):
            for j, value in enumerate(metric_values[name]):
                plt.text(x[j] + width*(i-2.5), value + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', 
                        rotation=75, fontsize=8)
        
        # 图表美化
        title = f'Model Performance Metrics - {class_name.upper()}' if class_name != 'all' else 'Model Performance Metrics - ALL (Average)'
        plt.title(title, pad=20, fontsize=14, y=1.02)
        plt.xlabel('Models', labelpad=10)
        plt.ylabel('Score', labelpad=10)
        plt.xticks(x, models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)  # 设置y轴范围
        
        # 调整图例位置
        plt.legend(bbox_to_anchor=(1.05, 1.02), loc='upper left',
                  borderaxespad=0.5, frameon=True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'performance_{class_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

# ==================== 主流程 ====================
def plot_performance_metrics(info_files, custom_labels, save_dir):
    """绘制并保存所有性能指标图"""
    # 收集数据
    metrics_data = {}
    
    for file_path, label in zip(info_files, custom_labels):
        # 解析性能指标
        model_metrics = parse_metrics_file(file_path)
        if model_metrics:
            metrics_data[label] = model_metrics
            print(f"已加载: {label} 的性能指标")
    
    if not metrics_data:
        print("没有有效数据可绘制！")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制性能指标图表（rp, tpt, tpl, brl, bim, all）
    class_names = ['rp', 'tpt', 'tpl', 'brl', 'bim', 'all']
    plot_metrics_comparison(metrics_data, class_names, save_dir)
    
    print("\n所有性能指标图表生成完成！")

if __name__ == '__main__':
    # 初始化字体配置
    configure_font()
    
    # 设置输出目录
    SAVE_DIR = "/root/RTDETR/plots/prune/try"
    print("开始生成性能指标图表...")
    plot_performance_metrics(info_files, custom_labels, SAVE_DIR)