import os
import random


def convert_yolo_to_yolov8_pose():
    input_dir = r"C:\Users\lifan\Desktop\XJTU-VMUP\VMUP2400datasets\train\labels"
    output_dir = r"C:\Users\lifan\Desktop\train\labels"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有输入文件
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 初始化关键点存储
        keypoints = []

        with open(input_path, 'r') as f:
            lines = f.readlines()

        # 解析原始标注，收集所有关键点
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                cls_id = int(parts[0])
                x_center = float(parts[1]) * 1080  # 转换为像素坐标
                y_center = float(parts[2]) * 1080
                keypoints.append((x_center, y_center))
            except:
                continue

        # 如果没有关键点则跳过
        if not keypoints:
            print(f"跳过文件 {filename}: 没有有效关键点")
            continue

        # 计算边界框
        x_coords = [kp[0] for kp in keypoints]
        y_coords = [kp[1] for kp in keypoints]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 生成随机扩展量（10-20像素）
        delta_x = random.randint(20, 40)
        delta_y = random.randint(20, 40)

        # 计算边界框（确保不超出图像范围）
        left = max(0, x_min - delta_x)
        right = min(1080, x_max + delta_x)
        top = max(0, y_min - delta_y)
        bottom = min(1080, y_max + delta_y)

        # 计算归一化参数
        box_x_center = (left + right) / 2 / 1080
        box_y_center = (top + bottom) / 2 / 1080
        box_width = (right - left) / 1080
        box_height = (bottom - top) / 1080

        # 构建关键点字符串（所有点按原始顺序）
        kp_str = []
        for x, y in keypoints:
            kp_str.extend([f"{x / 1080:.6f}", f"{y / 1080:.6f}", "2"])  # 可见性设为2

        # 生成YOLOv8-pose格式（只有一个palm目标框）
        yolo_line = f"0 {box_x_center:.6f} {box_y_center:.6f} {box_width:.6f} {box_height:.6f} {' '.join(kp_str)}"

        # 写入输出文件
        with open(output_path, 'w') as f:
            f.write(yolo_line + "\n")


if __name__ == "__main__":
    convert_yolo_to_yolov8_pose()