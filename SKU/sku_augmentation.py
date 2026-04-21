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
            001_001.jpg    # 面编号_增强编号
            001_002.jpg
            001_003.jpg
            001_004.jpg
            001_005.jpg
            002_001.jpg    # 第二张原图的增强
            ...
        000002/
            001_001.jpg
            ...
        metadata.json

固定增强方案（5张/每面）：

| 编号 | 增强类型     | 核心操作                          | 物理意义              |
|-----|-------------|----------------------------------|---------------------|
| 001 | 左侧视角     | perspective(左倾0.003)          | 左侧斜向拍摄视角       |
| 002 | 右侧视角     | perspective(右倾0.003)          | 右侧斜向拍摄视角       |
| 003 | 暗光环境     | hsv_v(-0.25), hsv_s(-0.15)       | 光线不足，颜色偏暗     |
| 004 | 亮光环境     | hsv_v(+0.25), hsv_s(+0.10)       | 强光照射，颜色偏亮     |
| 005 | 低质量模拟   | scale(0.9) + blur(3) + noise(5)  | 远距离/低质量拍摄     |

使用方法：
    python sku_augmentation.py --input ./sku_raw --output ./sku_library

作者：毕设项目
日期：2026年4月
"""

import os
import sys
import argparse
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm


# ============ 固定增强方案定义 ============

FIXED_AUGMENTATION_PLAN = [
    {
        'id': '001',
        'name': 'left_perspective',
        'name_cn': '左侧视角',
        'description': '透视变换模拟左侧斜向拍摄',
        'operations': [
            {'type': 'perspective', 'direction': 'left', 'strength': 0.003}
        ]
    },
    {
        'id': '002',
        'name': 'right_perspective',
        'name_cn': '右侧视角',
        'description': '透视变换模拟右侧斜向拍摄',
        'operations': [
            {'type': 'perspective', 'direction': 'right', 'strength': 0.003}
        ]
    },
    {
        'id': '003',
        'name': 'dim_environment',
        'name_cn': '暗光环境',
        'description': 'HSV空间调整：亮度-25%，饱和度-15%',
        'operations': [
            {'type': 'hsv', 'h': 0, 's': -0.15, 'v': -0.25}
        ]
    },
    {
        'id': '004',
        'name': 'bright_environment',
        'name_cn': '亮光环境',
        'description': 'HSV空间调整：亮度+25%，饱和度+10%',
        'operations': [
            {'type': 'hsv', 'h': 0, 's': 0.10, 'v': 0.25}
        ]
    },
    {
        'id': '005',
        'name': 'low_quality',
        'name_cn': '低质量模拟',
        'description': '缩放90%+模糊+噪声，模拟远距离拍摄',
        'operations': [
            {'type': 'scale', 'factor': 0.9},
            {'type': 'blur', 'kernel': 3},
            {'type': 'noise', 'std': 5}
        ]
    }
]


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
    
    images_per_face = len(FIXED_AUGMENTATION_PLAN)
    
    print("=" * 70)
    print("SKU图片库建设")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"SKU文件夹数: {len(sku_folders)}")
    print(f"每面增强数量: {images_per_face} 张")
    print("=" * 70)
    print("\n固定增强方案:")
    for plan in FIXED_AUGMENTATION_PLAN:
        print(f"  [{plan['id']}] {plan['name_cn']}: {plan['description']}")
    print("=" * 70)
    
    # 元数据
    metadata = {
        'created_at': datetime.now().isoformat(),
        'augmentation_plan': FIXED_AUGMENTATION_PLAN,
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
        sku_name = sku_folder.name  # 如 000001
        
        # 收集该SKU的所有图片（按文件名排序）
        face_images = []
        for ext in image_extensions:
            face_images.extend(sku_folder.glob(f'*{ext}'))
            face_images.extend(sku_folder.glob(f'*{ext.upper()}'))
        face_images = sorted(face_images, key=lambda x: x.name)
        
        if not face_images:
            stats['errors'].append(f"SKU {sku_name} 没有图片")
            continue
        
        sku_metadata = {
            'sku_name': sku_name,
            'source_folder': str(sku_folder),
            'faces': []
        }
        
        # 处理每张图片（面编号从001开始）
        for face_idx, face_img in enumerate(face_images, start=1):
            face_id = f"{face_idx:03d}"  # 001, 002, 003...
            
            # 读取图片
            image = cv2.imread(str(face_img))
            if image is None:
                stats['errors'].append(f"无法读取: {face_img}")
                continue
            
            stats['total_faces'] += 1
            
            # 对该面应用所有增强
            face_augmentations = []
            for plan in FIXED_AUGMENTATION_PLAN:
                aug_image = apply_augmentation_plan(image, plan)

                # 生成输出文件名：面编号_增强编号.jpg，保存到SKU文件夹
                output_name = f"{face_id}_{plan['id']}.jpg"
                sku_output_dir = output_path / sku_name
                sku_output_dir.mkdir(parents=True, exist_ok=True)
                output_path_file = sku_output_dir / output_name
                cv2.imwrite(str(output_path_file), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                stats['total_images'] += 1
                face_augmentations.append({
                    'aug_id': plan['id'],
                    'aug_name': plan['name_cn'],
                    'output_file': output_name
                })
            
            sku_metadata['faces'].append({
                'face_id': face_id,
                'source_file': face_img.name,
                'image_size': [image.shape[1], image.shape[0]],
                'augmentations': face_augmentations
            })
        
        metadata['skus'].append(sku_metadata)
    
    # 保存元数据
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
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
    {SKU编号}/{面编号}_{增强编号}.jpg
    例如：000001/001_001.jpg

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
