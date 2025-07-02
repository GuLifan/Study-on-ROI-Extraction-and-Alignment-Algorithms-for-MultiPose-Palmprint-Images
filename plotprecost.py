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
]

custom_labels = [
    'YOLOv8n', 'YOLOv8s', 'YOLOv8m',
    'YOLOv10n', 'YOLOv10s',
    'YOLO11n', 'YOLO11s',
    'RT-DETR-l', 'RT-DETR-R18', 'RT-DETR-R34', 'RT-DETR-R50'
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
        
        return {
            'GFLOPs': float(parts[0]),
            'Parameters': int(parts[1].replace(',', '')),
            'Preprocess': float(parts[2].replace('s', '')) * 1000,  # 转换为ms
            'Inference': float(parts[3].replace('s', '')) * 1000,    # 转换为ms
            'Postprocess': float(parts[4].replace('s', '')) * 1000,  # 转换为ms
            'FPS_total': float(parts[5]),
            'FPS_inference': float(parts[6]),
            'Model_Size': float(parts[7].replace('MB', ''))
        }
    except Exception as e:
        print(f"解析错误 {file_path}: {str(e)}")
        return None

# ==================== 绘图部分 ====================
def plot_combined_metrics(df, save_path):
    """绘制GFLOPs、Parameters和Model Size的复合图表"""
    # 配置字体
    configure_font()
    
    # 定义颜色方案
    colors = [
        '#205EA7',  # GFLOPs
        '#38B6C4',   # Parameters
        '#7ECBB9',   # Model Size
    ]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 调整顶部空间 - 这是控制顶部框线和柱子之间空隙的参数
    plt.subplots_adjust(top=0.98)  # 从0.8增加到0.85，增加了顶部空间
    
    # 设置x轴位置
    x = np.arange(len(df))
    width = 0.25
    
    # 绘制GFLOPs (左侧y轴)
    bars1 = ax1.bar(x - width, df['GFLOPs'], width, 
                   color=colors[0], label='GFLOPs')
    ax1.set_ylabel('GFLOPs', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # 添加GFLOPs数值标签（使用柱子颜色）
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45,
                fontsize=10, color=colors[0])
    
    # 创建第二个y轴 (右侧)
    ax2 = ax1.twinx()
    
    # 绘制Parameters (右侧y轴)
    bars2 = ax2.bar(x, df['Parameters']/1e6, width, 
                   color=colors[1], label='Parameters (M)')
    ax2.set_ylabel('Parameters (Millions)', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    # 添加Parameters数值标签（使用柱子颜色）
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=45,
                fontsize=10, color=colors[1])
    
    # 绘制Model Size (右侧y轴)
    bars3 = ax2.bar(x + width, df['Model_Size'], width,
                   color=colors[2], label='Model Size (MB)')
    
    # 添加Model Size数值标签（使用柱子颜色）
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45,
                fontsize=10, color=colors[2])
    
    # 图表美化
    ax1.set_title('Model Efficiency Metrics Comparison', pad=20, fontsize=14, y=1.02)  # 增加了y值
    ax1.set_xlabel('Models', labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # 合并图例并调整位置
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, 
              bbox_to_anchor=(1.05, 1.02), loc='upper left',
              borderaxespad=0.5, frameon=True)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

def plot_time_metrics(df, save_path):
    """绘制时间指标图表(单位: ms)"""
    # 配置字体
    configure_font()
    
    # 定义颜色方案
    colors = ['#EAF0B2', '#3BB6C5', '#C8E2B2']
    
    plt.figure(figsize=(14, 7))
    # 调整顶部空间 - 这是控制顶部框线和柱子之间空隙的参数
    plt.subplots_adjust(top=0.98)  # 从0.8增加到0.85，增加了顶部空间
    
    bar_width = 0.25
    index = np.arange(len(df))
    
    # 绘制每组柱状图
    bars1 = plt.bar(index - bar_width, df['Preprocess'], 
                   width=bar_width, color=colors[0], label='Preprocess (ms)')
    bars2 = plt.bar(index, df['Inference'], 
                   width=bar_width, color=colors[1], label='Inference (ms)')
    bars3 = plt.bar(index + bar_width, df['Postprocess'], 
                   width=bar_width, color=colors[2], label='Postprocess (ms)')
    
    # 添加数值标签（使用各自柱子的颜色）
    for bar, color in zip(bars1, [colors[0]]*len(bars1)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                height * 1.01,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45, 
                fontsize=10, color=color)
    
    for bar, color in zip(bars2, [colors[1]]*len(bars2)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                height * 1.01,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45, 
                fontsize=10, color=color)
    
    for bar, color in zip(bars3, [colors[2]]*len(bars3)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                height * 1.01,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45, 
                fontsize=10, color=color)
    
    # 图表美化
    plt.title('Inference Time Components (ms)', pad=20, fontsize=14, y=1.02)  # 增加了y值
    plt.xlabel('Models', labelpad=10)
    plt.ylabel('Time (milliseconds)', labelpad=10)
    plt.xticks(index, df['Model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc='upper left',
              borderaxespad=0.5, frameon=True)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

def plot_fps_metrics(df, save_path):
    """绘制FPS指标图表"""
    # 配置字体
    configure_font()
    
    # 定义颜色方案
    colors = ['#965478', '#1D91C2']
    
    plt.figure(figsize=(14, 7))
    # 调整顶部空间 - 这是控制顶部框线和柱子之间空隙的参数
    plt.subplots_adjust(top=0.98)  # 从0.8增加到0.85，增加了顶部空间
    
    bar_width = 0.35
    index = np.arange(len(df))
    
    # 绘制FPS柱状图
    bars1 = plt.bar(index - bar_width/2, df['FPS_total'], 
                   width=bar_width, color=colors[0], label='FPS (total)')
    bars2 = plt.bar(index + bar_width/2, df['FPS_inference'], 
                   width=bar_width, color=colors[1], label='FPS (inference)')
    
    # 添加数值标签（使用各自柱子的颜色）
    for bar, color in zip(bars1, [colors[0]]*len(bars1)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                height * 1.01,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45, 
                fontsize=10, color=color)
    
    for bar, color in zip(bars2, [colors[1]]*len(bars2)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                height * 1.01,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=45, 
                fontsize=10, color=color)
    
    # 图表美化
    plt.title('FPS Performance Comparison', pad=20, fontsize=14, y=1.02)  # 增加了y值
    plt.xlabel('Models', labelpad=10)
    plt.ylabel('Frames Per Second (FPS)', labelpad=10)
    plt.xticks(index, df['Model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.05, 1.02), loc='upper left',
              borderaxespad=0.5, frameon=True)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

# ==================== 主流程 ====================
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
    
    # 绘制复合图表
    plot_combined_metrics(df, os.path.join(save_dir, 'combined_metrics.png'))
    
    # 绘制时间指标图表
    plot_time_metrics(df, os.path.join(save_dir, 'time_metrics_ms.png'))
    
    # 绘制FPS指标图表
    plot_fps_metrics(df, os.path.join(save_dir, 'fps_metrics.png'))
    
    # 保存原始数据
    data_path = os.path.join(save_dir, 'metrics_data.csv')
    df.to_csv(data_path, index=False)
    print(f"\n原始数据已保存: {data_path}")

if __name__ == '__main__':
    # 初始化字体配置
    configure_font()
    
    # 设置输出目录
    SAVE_DIR = "/root/RTDETR/plots/111"
    print("开始生成图表...")
    plot_all_metrics(info_files, custom_labels, SAVE_DIR)
    print("\n所有图表生成完成！")