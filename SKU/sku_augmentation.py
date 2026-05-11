"""
SKU图片库增强脚本

参考：
- YOLOv8/YOLO11 默认增强参数

输入目录结构：
    input_dir/
        000001/
            IMG_001.jpg
            IMG_002.jpg
            ...
        000002/
            photo1.png
            photo2.png
            ...
        ...

输出目录结构：
    output_dir/
        000001/
            IMG_001_aug01.jpg    # 随机增强1
            IMG_001_aug02.jpg    # 随机增强2
            IMG_001_rot90.jpg    # 旋转90度
            IMG_001_rot180.jpg   # 旋转180度
            IMG_001_rot270.jpg   # 旋转270度
            IMG_002_aug01.jpg ~ IMG_002_rot270.jpg  # 第二张原图的增强
            ...
        000002/
            photo1_aug01.jpg ~ photo1_rot270.jpg
            ...
        metadata.json

增强策略（5张/每面）：

| 类型 | 数量 | 输出格式       | 核心操作                          | 物理意义                      |
|-----|------|---------------|----------------------------------|-----------------------------|
| 随机增强 | 2    | aug01, aug02  | 随机裁剪(50%) + 随机擦除(50%)    | 模拟检测框裁剪差异和遮挡      |
| 旋转增强 | 3    | rot90, rot180, rot270 | 旋转90/180/270度            | 模拟箱体不同朝向              |

随机增强说明：
- random_crop_resize: 随机裁剪60%-95%区域后resize，模拟检测框裁剪比例差异
- random_erasing: 随机擦除10%-20%区域（用周围均值填充），模拟遮挡

使用方法：
    python sku_augmentation.py --input ./sku_raw --output ./sku_library

作者：毕设项目
日期：2026年4月
"""

import os
import sys
import argparse
import json
import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm


# ============ 增强策略 ============
NUM_RANDOM_AUGS = 2  # 每张原图生成2次随机增强
ROTATION_ANGLES = [90, 180, 270]  # 旋转增强（建库用，不参与训练）


def apply_random_augmentation(image: np.ndarray) -> np.ndarray:
    """随机组合增强：随机裁剪 + 随机擦除"""
    result = image.copy()
    # 50%概率做随机裁剪
    if np.random.random() < 0.5:
        result = random_crop_resize(result)
    # 50%概率做随机擦除
    if np.random.random() < 0.5:
        result = random_erasing(result)
    return result


# ============ 增强函数 ============

def apply_perspective_transform(image: np.ndarray, 
                                  direction: str, 
                                  strength: float) -> np.ndarray:
    """透视变换（模拟不同拍摄角度）"""
    h, w = image.shape[:2]
    offset = int(min(h, w) * strength * 10)
    
    src_points = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    
    if direction == 'left':
        dst_points = np.float32([
            [offset, offset],
            [w - offset//2, offset],
            [w - offset//2, h - offset//2],
            [offset, h - offset]
        ])
    else:
        dst_points = np.float32([
            [offset//2, offset],
            [w - offset, offset],
            [w - offset, h - offset],
            [offset//2, h - offset//2]
        ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT_101)
    return result


def adjust_hsv(image: np.ndarray, 
               h_shift: float, 
               s_shift: float, 
               v_shift: float) -> np.ndarray:
    """HSV空间颜色增强"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift * 180) % 180
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + s_shift)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + v_shift)
    
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def scale_image(image: np.ndarray, factor: float) -> np.ndarray:
    """缩放图像"""
    h, w = image.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    
    if factor < 1:
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        result = np.full_like(image, 128)
        result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled
    else:
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        result = scaled[start_y:start_y+h, start_x:start_x+w]
    
    return result


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """高斯模糊"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_gaussian_noise(image: np.ndarray, std: int) -> np.ndarray:
    """添加高斯噪声"""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """旋转图像90/180/270度"""
    h, w = image.shape[:2]
    
    if angle == 90:
        # 顺时针旋转90度
        rotated = cv2.transpose(image)
        rotated = cv2.flip(rotated, 1)
    elif angle == 180:
        # 旋转180度
        rotated = cv2.flip(image, -1)
    elif angle == 270:
        # 顺时针旋转270度 = 逆时针90度
        rotated = cv2.transpose(image)
        rotated = cv2.flip(rotated, 0)
    else:
        # 其他角度使用标准旋转（但保持尺寸）
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    return rotated


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """调整对比度"""
    # 转换为浮点数
    img_float = image.astype(np.float32)
    # 计算均值
    mean = np.mean(img_float, axis=(0, 1), keepdims=True)
    # 调整对比度：new = mean + factor * (old - mean)
    adjusted = mean + factor * (img_float - mean)
    # 裁剪到有效范围
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def random_crop_resize(image: np.ndarray, min_area_ratio: float = 0.6, max_area_ratio: float = 0.95) -> np.ndarray:
    """随机裁剪+resize，模拟检测框裁剪比例差异和局部可见"""
    h, w = image.shape[:2]
    area_ratio = np.random.uniform(min_area_ratio, max_area_ratio)
    # 随机宽高比偏移
    aspect_ratio = np.random.uniform(0.7, 1.3)
    
    crop_area = h * w * area_ratio
    crop_h = int(np.sqrt(crop_area / aspect_ratio))
    crop_w = int(np.sqrt(crop_area * aspect_ratio))
    
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    
    y = np.random.randint(0, h - crop_h + 1)
    x = np.random.randint(0, w - crop_w + 1)
    
    cropped = image[y:y+crop_h, x:x+crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def random_erasing(image: np.ndarray, min_area: float = 0.1, max_area: float = 0.2) -> np.ndarray:
    """随机擦除，模拟遮挡和局部残缺，迫使模型学全局语义"""
    result = image.copy()
    h, w = image.shape[:2]
    
    erase_area = np.random.uniform(min_area, max_area) * h * w
    aspect_ratio = np.random.uniform(0.5, 2.0)
    
    erase_h = int(np.sqrt(erase_area / aspect_ratio))
    erase_w = int(np.sqrt(erase_area * aspect_ratio))
    
    erase_h = min(erase_h, h)
    erase_w = min(erase_w, w)
    
    y = np.random.randint(0, h - erase_h + 1)
    x = np.random.randint(0, w - erase_w + 1)
    
    # 用周围区域均值填充（比随机噪声更真实）
    patch = image[max(0,y-5):y+erase_h+5, max(0,x-5):x+erase_w+5]
    fill_value = patch.mean(axis=(0,1)).astype(np.uint8)
    result[y:y+erase_h, x:x+erase_w] = fill_value
    
    return result


def apply_single_operation(image: np.ndarray, operation: Dict) -> np.ndarray:
    """应用单个增强操作"""
    op_type = operation['type']
    
    if op_type == 'perspective':
        return apply_perspective_transform(image, operation['direction'], operation['strength'])
    elif op_type == 'hsv':
        return adjust_hsv(image, operation['h'], operation['s'], operation['v'])
    elif op_type == 'scale':
        return scale_image(image, operation['factor'])
    elif op_type == 'blur':
        return apply_gaussian_blur(image, operation['kernel'])
    elif op_type == 'noise':
        return add_gaussian_noise(image, operation['std'])
    elif op_type == 'rotate':
        return rotate_image(image, operation['angle'])
    elif op_type == 'contrast':
        return adjust_contrast(image, operation['factor'])
    else:
        raise ValueError(f"未知操作类型: {op_type}")


def apply_augmentation_plan(image: np.ndarray, plan: Dict) -> np.ndarray:
    """应用增强方案"""
    result = image.copy()
    for operation in plan['operations']:
        result = apply_single_operation(result, operation)
    return result


# ============ SKU库构建 ============

def build_sku_library(input_dir: str, output_dir: str) -> Dict:
    """
    构建SKU图片库
    
    输入目录结构：
        input_dir/
            000001/
                IMG_001.jpg
                IMG_002.jpg
            000002/
                photo1.png
                ...
    
    输出命名规则：
        {SKU编号}/{面编号}_{增强编号}.jpg
        例如：000001/001_001.jpg
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 固定随机种子
    np.random.seed(42)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 收集所有SKU文件夹（按名称排序，确保顺序一致）
    sku_folders = sorted([d for d in input_path.iterdir() if d.is_dir()], 
                         key=lambda x: x.name)
    
    images_per_face = NUM_RANDOM_AUGS + len(ROTATION_ANGLES)
    
    print("=" * 70)
    print("SKU图片库建设")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"SKU文件夹数: {len(sku_folders)}")
    print(f"每面增强数量: {images_per_face} 张 ({NUM_RANDOM_AUGS}随机 + {len(ROTATION_ANGLES)}旋转)")
    print("=" * 70)
    print("\n增强策略:")
    print(f"  - 随机增强: {NUM_RANDOM_AUGS}次/图（随机裁剪+随机擦除）")
    print(f"  - 旋转增强: {ROTATION_ANGLES}度")
    print("=" * 70)
    
    # 读取 sku_database.json 获取 sku_name 映射
    sku_name_map = {}
    db_path = input_path / 'sku_database.json'
    if db_path.exists():
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            for sku_id, info in db.items():
                if isinstance(info, dict):
                    sku_name_map[sku_id] = info.get('name', sku_id)
                else:
                    sku_name_map[sku_id] = sku_id
            print(f"  ✓ 已加载 {len(sku_name_map)} 个SKU的命名信息")
        except Exception as e:
            print(f"  ⚠ 无法读取 sku_database.json: {e}")
    
    # 元数据
    metadata = {
        'created_at': datetime.now().isoformat(),
        'augmentation_plan': {
            'num_random_augs': NUM_RANDOM_AUGS,
            'rotation_angles': ROTATION_ANGLES,
            'description': '随机增强(随机裁剪+随机擦除) + 旋转增强'
        },
        'reference': 'YOLOv8/YOLO11 augmentation strategy',
        'total_skus': len(sku_folders),
        'images_per_face': images_per_face,
        'skus': []
    }
    
    stats = {
        'total_skus': len(sku_folders),
        'total_faces': 0,
        'total_images': 0,
        'errors': []
    }
    
    for sku_folder in tqdm(sku_folders, desc="处理SKU"):
        sku_id = sku_folder.name  # 如 000001
        sku_name = sku_name_map.get(sku_id, sku_id)  # 从数据库获取真实名称
        
        # 收集该SKU的所有图片（按文件名排序），使用set去重
        face_images_set = set()
        for ext in image_extensions:
            face_images_set.update(sku_folder.glob(f'*{ext}'))
            face_images_set.update(sku_folder.glob(f'*{ext.upper()}'))
        face_images = sorted(face_images_set, key=lambda x: x.name)
        
        if not face_images:
            stats['errors'].append(f"SKU {sku_name} 没有图片")
            continue
        
        sku_metadata = {
            'sku_id': sku_id,
            'sku_name': sku_name,
            'source_folder': str(sku_folder),
            'faces': []
        }
        
        # 处理每张图片
        for face_idx, face_img in enumerate(face_images, start=1):
            # 使用原图文件名（去扩展名）作为前缀
            face_stem = face_img.stem  # 如 "IMG_001"
            
            # 读取图片
            image = cv2.imread(str(face_img))
            if image is None:
                stats['errors'].append(f"无法读取: {face_img}")
                continue
            
            stats['total_faces'] += 1
            
            # 对该面应用所有增强
            face_augmentations = []
            aug_idx = 0
            # 随机增强（用于训练）
            for i in range(NUM_RANDOM_AUGS):
                aug_image = apply_random_augmentation(image)
                aug_idx += 1
                output_name = f"{face_stem}_aug{aug_idx:02d}.jpg"
                sku_output_dir = output_path / sku_id
                sku_output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sku_output_dir / output_name), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats['total_images'] += 1
                face_augmentations.append({
                    'aug_id': f'aug{aug_idx:02d}',
                    'aug_name': '随机增强',
                    'output_file': output_name
                })

            # 旋转增强（用于建库匹配）
            for angle in ROTATION_ANGLES:
                aug_image = rotate_image(image, angle)
                aug_idx += 1
                output_name = f"{face_stem}_rot{angle}.jpg"
                cv2.imwrite(str(sku_output_dir / output_name), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats['total_images'] += 1
                face_augmentations.append({
                    'aug_id': f'rot{angle}',
                    'aug_name': f'旋转{angle}度',
                    'output_file': output_name
                })
            
            sku_metadata['faces'].append({
                'face_id': str(face_idx),
                'source_file': face_img.name,
                'source_stem': face_stem,
                'image_size': [image.shape[1], image.shape[0]],
                'augmentations': face_augmentations
            })
        
        metadata['skus'].append(sku_metadata)
    
    # 保存元数据
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 生成 sku_library.csv 索引文件
    # 字段: image_name,sku_id,label,sku_name
    # label格式: SKU编号，同一SKU不同面共享label
    csv_rows = []
    for sku_metadata in metadata['skus']:
        sku_id = sku_metadata['sku_id']
        sku_name = sku_metadata['sku_name']
        label = sku_id  # label = SKU编号，同一SKU不同面共享label
        for face in sku_metadata['faces']:
            for aug in face['augmentations']:
                image_name = f"{sku_id}/{aug['output_file']}"
                csv_rows.append({
                    'image_name': image_name,
                    'sku_id': sku_id,
                    'label': label,
                    'sku_name': sku_name
                })
    
    # 写入CSV
    csv_path = output_path / 'sku_library.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'sku_id', 'label', 'sku_name'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"已生成索引文件: {csv_path}")
    
    # 打印统计
    print("\n" + "=" * 70)
    print("构建完成")
    print("=" * 70)
    print(f"SKU数量: {stats['total_skus']}")
    print(f"总原始图片(面数): {stats['total_faces']}")
    print(f"总输出图片: {stats['total_images']}")
    print(f"输出目录: {output_path}")
    if stats['errors']:
        print(f"错误数: {len(stats['errors'])}")
        for err in stats['errors'][:5]:
            print(f"  - {err}")
    print("=" * 70)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='SKU图片库增强脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输入目录结构：
    sku_raw/
        000001/
            IMG_001.jpg
            IMG_002.jpg
            ...
        000002/
            photo1.png
            photo2.png
            ...

输出命名规则：
    {SKU编号}/{原图名}_{增强编号}.jpg
    例如：000001/IMG_001_001.jpg

示例命令：
    python sku_augmentation.py --input ./sku_output --output ./sku_library
        """
    )
    parser.add_argument('--input', type=str, required=True, 
                        help='输入SKU根目录（包含各SKU子文件夹）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出SKU库目录')
    
    args = parser.parse_args()
    
    stats = build_sku_library(input_dir=args.input, output_dir=args.output)
    
    return 0 if not stats['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
