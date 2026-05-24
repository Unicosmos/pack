#!/usr/bin/env python3
"""
标注可视化工具
支持 COCO 和 YOLO 格式的分割标注可视化

Usage:
    python visualize_annotations.py -i <image_path> -l <label_path> -o <output_dir> [--format <coco|yolo>]
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from PIL import Image, ImageDraw, ImageFont


# 类别颜色 (4个类别)
CLASS_COLORS = [
    (255, 0, 0),      # 红色 - 类别0
    (0, 255, 0),      # 绿色 - 类别1
    (0, 0, 255),      # 蓝色 - 类别2
    (255, 255, 0),    # 黄色 - 类别3
    (255, 0, 255),    # 品红 - 类别4
]

CLASS_NAMES = {
    0: "Carton-inner-all",
    1: "Carton-inner-occlusion",
    2: "Carton-outer-all",
    3: "Carton-outer-occlusion",
    4: "Unknown"
}


def get_font(size: int = 20) -> ImageFont.ImageFont:
    """获取可用字体"""
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    return font


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: List[float],
    color: Tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont
) -> None:
    """绘制单个多边形"""
    # 绘制多边形轮廓
    draw.polygon(polygon, outline=color, fill=None, width=3)
    
    # 计算多边形中心点用于放置标签
    if len(polygon) >= 4:
        x_coords = polygon[::2]
        y_coords = polygon[1::2]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # 绘制标签背景
        text_bbox = draw.textbbox((center_x, center_y), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        bg_x1 = center_x - text_width / 2 - 5
        bg_y1 = center_y - text_height / 2 - 5
        bg_x2 = center_x + text_width / 2 + 5
        bg_y2 = center_y + text_height / 2 + 5
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
        draw.text((center_x - text_width / 2, center_y - text_height / 2), 
                 label, fill=(255, 255, 255), font=font)


def load_coco_annotations(
    label_path: str,
    image_path: str
) -> List[Dict[str, Any]]:
    """
    加载 COCO 格式标注
    
    Args:
        label_path: COCO JSON文件路径
        image_path: 对应图片路径
        
    Returns:
        标注列表
    """
    with open(label_path, 'r') as f:
        coco_data = json.load(f)
    
    # 获取图片文件名
    image_name = Path(image_path).name
    
    # 查找对应的图片ID
    image_id = None
    for img in coco_data.get('images', []):
        if img.get('file_name') == image_name:
            image_id = img.get('id')
            img_width = img.get('width')
            img_height = img.get('height')
            break
    
    if image_id is None:
        # 尝试直接从文件名提取
        print(f"Warning: Could not find image {image_name} in COCO annotations")
        return []
    
    # 查找对应的标注
    annotations = []
    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') == image_id:
            annotations.append(ann)
    
    return annotations


def load_yolo_annotations(
    label_path: str,
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:
    """
    加载 YOLO 格式标注
    
    Args:
        label_path: YOLO txt文件路径
        image_width: 图片宽度
        image_height: 图片高度
        
    Returns:
        标注列表
    """
    annotations = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = list(map(float, line.split()))
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            coords = parts[1:]
            
            # 反归一化坐标
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * image_width
                    y = coords[i + 1] * image_height
                    polygon.extend([x, y])
            
            annotations.append({
                'category_id': class_id,
                'segmentation': [polygon]
            })
    
    return annotations


def visualize_image(
    image_path: str,
    label_path: str,
    output_dir: str,
    format_type: str = 'coco'
) -> str:
    """
    可视化单张图片
    
    Args:
        image_path: 图片路径
        label_path: 标注路径
        output_dir: 输出目录
        format_type: 标注格式 ('coco' 或 'yolo')
        
    Returns:
        输出图片路径
    """
    # 加载图片
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_width, img_height = image.size
    
    # COCO到YOLO的类别映射
    coco_to_yolo = {1: 0, 2: 1, 3: 2, 4: 3}
    
    # 加载标注
    if format_type == 'coco':
        annotations = load_coco_annotations(label_path, image_path)
    elif format_type == 'yolo':
        annotations = load_yolo_annotations(label_path, img_width, img_height)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # 创建绘制对象
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    font = get_font(16)
    
    # 绘制每个标注
    for ann in annotations:
        category_id = ann.get('category_id', 0)
        
        # 根据格式类型进行类别ID映射
        if format_type == 'coco':
            yolo_id = coco_to_yolo.get(category_id, category_id - 1)
        else:
            yolo_id = category_id
        
        color = CLASS_COLORS[yolo_id % len(CLASS_COLORS)]
        class_name = CLASS_NAMES.get(yolo_id, f"Class {yolo_id}")
        
        segmentation = ann.get('segmentation', [])
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if polygon:
                    label = f"{class_name}"
                    draw_polygon(draw, polygon, color, label, font)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"vis_{Path(image_path).name}"
    output_path = os.path.join(output_dir, output_name)
    result_image.save(output_path)
    
    print(f"Visualized: {image_path} -> {output_path}")
    print(f"  - Annotations: {len(annotations)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='标注可视化工具')
    parser.add_argument('-i', '--image', required=True, help='输入图片路径')
    parser.add_argument('-l', '--label', required=True, help='输入标注路径')
    parser.add_argument('-o', '--output', required=True, help='输出目录')
    parser.add_argument('--format', choices=['coco', 'yolo'], default='coco', 
                       help='标注格式 (default: coco)')
    
    args = parser.parse_args()
    
    try:
        output_path = visualize_image(
            args.image,
            args.label,
            args.output,
            args.format
        )
        print(f"\n成功！输出文件: {output_path}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
