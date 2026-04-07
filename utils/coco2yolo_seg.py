"""
COCO格式转YOLO格式脚本（支持检测和分割）
适用于LSCD数据集

YOLO分割格式说明:
- 每张图片对应一个.txt标注文件
- 每行格式: class_id x1 y1 x2 y2 x3 y3 ... (归一化多边形坐标)
- 目录结构:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo(
    coco_json_path: str,
    images_src_dir: str,
    output_dir: str,
    split: str = "train"
):
    """
    将COCO格式转换为YOLO格式（支持分割）
    
    Args:
        coco_json_path: COCO标注文件路径
        images_src_dir: 源图片目录
        output_dir: 输出目录
        split: 数据集划分 (train/val)
    """
    print(f"\n处理 {split} 集...")
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images_info = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"类别: {categories}")
    print(f"图片数量: {len(coco_data['images'])}")
    print(f"标注数量: {len(coco_data['annotations'])}")
    
    output_images_dir = Path(output_dir) / "images" / split
    output_labels_dir = Path(output_dir) / "labels" / split
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    category_id_to_yolo = {}
    for idx, (cat_id, cat_name) in enumerate(sorted(categories.items())):
        category_id_to_yolo[cat_id] = idx
    
    print(f"类别映射 (COCO ID -> YOLO ID): {category_id_to_yolo}")
    
    converted_count = 0
    skipped_count = 0
    
    for img_info in tqdm(coco_data['images'], desc=f"转换{split}集"):
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = img_info['file_name']
        
        src_img_path = Path(images_src_dir) / file_name
        
        if not src_img_path.exists():
            skipped_count += 1
            continue
        
        dst_img_path = output_images_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)
        
        anns = annotations_by_image.get(img_id, [])
        
        label_file_name = Path(file_name).stem + ".txt"
        label_path = output_labels_dir / label_file_name
        
        with open(label_path, 'w') as f:
            for ann in anns:
                yolo_class_id = category_id_to_yolo[ann['category_id']]
                
                segmentation = ann.get('segmentation', [])
                
                if segmentation and len(segmentation) > 0 and isinstance(segmentation[0], list):
                    for seg in segmentation:
                        if len(seg) >= 6:
                            normalized_seg = []
                            for i in range(0, len(seg), 2):
                                x_norm = max(0, min(1, seg[i] / img_width))
                                y_norm = max(0, min(1, seg[i + 1] / img_height))
                                normalized_seg.extend([x_norm, y_norm])
                            
                            seg_str = ' '.join([f'{v:.6f}' for v in normalized_seg])
                            f.write(f"{yolo_class_id} {seg_str}\n")
                else:
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        converted_count += 1
    
    print(f"转换完成: {converted_count} 张图片")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 张图片（找不到源文件）")
    
    return converted_count, skipped_count


def create_dataset_yaml(output_dir: str, class_names: list):
    """
    创建YOLO数据集配置文件
    
    Args:
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    yaml_content = f"""# LSCD数据集 - YOLO格式
# 用于地堆箱货识别和SKU匹配研究

path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# 类别数量
nc: {len(class_names)}

# 类别名称
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n数据集配置文件已创建: {yaml_path}")


def main():
    base_dir = "/root/source/data2/hyg/projects/pack/coco_style_oneclass"
    output_dir = "/root/source/data2/hyg/projects/pack/coco_style_oneclass/yolo_dataset_seg"
    
    train_json = os.path.join(base_dir, "annotations", "instances_train2017.json")
    val_json = os.path.join(base_dir, "annotations", "instances_val2017.json")
    train_images = os.path.join(base_dir, "yolo_dataset", "images", "train")
    val_images = os.path.join(base_dir, "yolo_dataset", "images", "val")
    
    print("=" * 60)
    print("COCO格式 -> YOLO格式 转换工具（支持分割）")
    print("=" * 60)
    print(f"源数据集目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    
    train_converted, train_skipped = convert_coco_to_yolo(
        train_json, train_images, output_dir, "train"
    )
    
    val_converted, val_skipped = convert_coco_to_yolo(
        val_json, val_images, output_dir, "val"
    )
    
    create_dataset_yaml(output_dir, ["Carton"])
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"训练集: {train_converted} 张图片")
    print(f"验证集: {val_converted} 张图片")
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  └── dataset.yaml")
    print("\n使用方法:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n-seg.pt')")
    print(f"  model.train(data='{output_dir}/dataset.yaml', epochs=100)")


if __name__ == "__main__":
    main()
