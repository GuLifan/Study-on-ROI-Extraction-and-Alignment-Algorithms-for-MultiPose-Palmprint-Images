import os
import shutil
import random


def copy_random_images(source_root, target_root, num_images=30):
    """
    从源目录的每个子文件夹下名为NC的文件夹中随机选取指定数量的图片复制到目标目录

    参数:
    source_root: 源根目录路径（如：C:\\10086）
    target_root: 目标根目录路径（如：C:\\10087）
    num_images: 要复制的图片数量（默认30）
    """
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # 遍历源目录中的每个子文件夹
    for folder_name in os.listdir(source_root):
        source_folder = os.path.join(source_root, folder_name)
        nc_folder = os.path.join(source_folder, 'NC')

        # 检查是否是有效目录且包含NC子目录
        if not os.path.isdir(source_folder) or not os.path.exists(nc_folder):
            continue

        # 收集所有图片文件路径
        all_images = []
        for filename in os.listdir(nc_folder):
            if filename.lower().endswith(image_extensions):
                all_images.append(os.path.join(nc_folder, filename))

        if not all_images:
            print(f"跳过空文件夹：{nc_folder}")
            continue

        # 随机选择图片（不超过实际数量）
        selected = random.sample(all_images, min(num_images, len(all_images)))

        # 创建目标目录（保持原文件夹结构）
        target_folder = os.path.join(target_root, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        # 复制选中的图片
        count = 0
        for img_src in selected:
            img_name = f"{folder_name}_{os.path.basename(img_src)}"
            img_dest = os.path.join(target_folder, img_name)
            shutil.copy2(img_src, img_dest)
            count += 1

        print(f"已从 {folder_name} 复制 {count} 张图片到 {target_folder}")


if __name__ == "__main__":
    # 配置路径参数
    SOURCE_DIR = r"C:\10086"
    TARGET_DIR = r"C:\10087"

    # 执行复制操作
    copy_random_images(SOURCE_DIR, TARGET_DIR)

    print("\n所有操作已完成！")