import os
import json
import shutil
from PIL import Image

# ================== 硬编码配置区域 ==================
yolo_root = r"C:\Users\lifan\Desktop\VMUP2400datasetsYOLO"  # YOLO数据集根目录
coco_root = r"C:\Users\lifan\Desktop\VMUP2400datasetsCOCO"  # COCO输出数据集根目录
class_list = ["rp", "tpt","tpl","brl","bim"]  # 类别列表（按YOLO的类别顺序排列）
splits = ["train", "val", "test"]  # 需要处理的数据集划分


# ================================================

def create_coco_dataset():
    # 创建COCO目录结构
    os.makedirs(os.path.join(coco_root, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(coco_root, "images"), exist_ok=True)

    # 生成类别信息
    categories = [{"id": i, "name": name} for i, name in enumerate(class_list)]

    for split in splits:
        # 初始化COCO数据结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        annotation_id = 0
        yolo_img_dir = os.path.join(yolo_root, "images", split)
        yolo_label_dir = os.path.join(yolo_root, "labels", split)

        # 遍历所有图像
        for img_idx, filename in enumerate(os.listdir(yolo_img_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # 原始图像路径
            src_img_path = os.path.join(yolo_img_dir, filename)

            try:
                with Image.open(src_img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"跳过损坏图像 {filename}: {str(e)}")
                continue

            # 目标图像路径（复制到COCO目录）
            dst_img_path = os.path.join(coco_root, "images", filename)
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)

            # 添加图像信息
            coco_data["images"].append({
                "id": img_idx,
                "file_name": filename,
                "width": width,
                "height": height
            })

            # 处理对应标签
            label_path = os.path.join(yolo_label_dir, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            for line in lines:
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                # 解析YOLO格式
                category_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # 转换为COCO格式
                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                w_abs = w * width
                h_abs = h * height

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": category_id,
                    "bbox": [x, y, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0
                })
                annotation_id += 1

        # 保存标注文件
        output_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"已生成 {split} 数据集，包含：")
        print(f"  - 图像数量：{len(coco_data['images'])}")
        print(f"  - 标注数量：{len(coco_data['annotations'])}")
        print(f"  - 保存路径：{output_path}\n")


if __name__ == "__main__":
    create_coco_dataset()
    print("COCO数据集生成完成！最终目录结构：")
    print(f"""
{coco_root}/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...""")