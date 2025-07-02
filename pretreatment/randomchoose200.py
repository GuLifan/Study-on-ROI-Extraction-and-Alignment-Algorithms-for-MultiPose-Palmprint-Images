import os
import random
import shutil

# 配置路径
source_img_dir = r"C:\Users\lifan\Desktop\test\images"
dest_img_dir = r"C:\Users\lifan\Desktop\images10"
source_txt_dir = r"C:\Users\lifan\Desktop\test\labels"
dest_txt_dir = r"C:\Users\lifan\Desktop\labels10"

# 创建目标目录（如果不存在）
os.makedirs(dest_img_dir, exist_ok=True)
os.makedirs(dest_txt_dir, exist_ok=True)

# 获取所有jpg文件
all_images = [f for f in os.listdir(source_img_dir) if f.lower().endswith('.jpg')]

# 随机选择200个文件（如果不足200则全选）
selected_images = random.sample(all_images, min(10, len(all_images)))

# 移动文件
moved_count = 0
for img_name in selected_images:
    try:
        # 移动图片
        img_src = os.path.join(source_img_dir, img_name)
        img_dest = os.path.join(dest_img_dir, img_name)
        shutil.copy(img_src, img_dest)

        # 移动对应的txt文件
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_src = os.path.join(source_txt_dir, txt_name)
        txt_dest = os.path.join(dest_txt_dir, txt_name)

        if os.path.exists(txt_src):
            shutil.copy(txt_src, txt_dest)

        moved_count += 1
    except Exception as e:
        print(f"移动 {img_name} 时出错: {str(e)}")

print(f"成功移动 {moved_count} 对文件")