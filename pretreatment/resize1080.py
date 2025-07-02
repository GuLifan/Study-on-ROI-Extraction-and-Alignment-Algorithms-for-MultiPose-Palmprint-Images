import cv2
import mediapipe as mp
import os
from pathlib import Path

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 定义输入输出路径
input_folder  = r"C:\Users\lifan\Desktop\31-40"
output_folder = r"C:\Users\lifan\Desktop\31-40resized"

# 创建输出文件夹
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 处理每张图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {filename}")
            continue

        # 获取图片尺寸并验证
        h, w = img.shape[:2]
        if w != 1920 or h != 1080:
            print(f"图片尺寸不符: {filename} ({w}x{h})")
            continue

        # 进行手部检测
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            print(f"未检测到手部: {filename}")
            continue

        # 获取第一个检测到的手部关键点
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)

        # 计算裁剪区域
        x_center = (x_min + x_max) / 2
        allowed_start_min = max(x_max - 1080, 0)
        allowed_start_max = min(x_min, 840)  # 1920 - 1080 = 840

        if allowed_start_min > allowed_start_max:
            # 当手部区域过宽时居中裁剪
            x_start = max(0, min(int(x_center - 540), 840))
        else:
            # 优化裁剪位置确保包含整个手部
            x_start = x_center - 540
            x_start = max(allowed_start_min, min(x_start, allowed_start_max))

        x_start = int(max(0, min(x_start, 840)))

        # 执行裁剪
        cropped_img = img[0:1080, x_start:x_start + 1080]

        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped_img)
        print(f"已处理: {filename}")

# 释放资源
hands.close()