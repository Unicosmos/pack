"""
SKU特征建库脚本

从 sku_output/ 读取分类好的SKU图片，
使用 ViT-S16 DINO 提取384维特征，
输出到 sku_library/ 目录供Web后端使用。

支持两种模式：
1. 直接模式：从 sku_output/ 读取原始图片
2. 增强模式：从 sku_augmentation.py 的输出读取增强图片

输出文件:
- sku_library/sku_features.npy  [N, 384] 特征矩阵
- sku_library/sku_library.csv   索引文件，包含 image_name,sku_id,label,sku_name

使用方法:
    python build_library.py --input ./sku_output --output ../sku_library --model-path ./models/sku_trained_vits16_dino.pth
    python build_library.py --device cuda  # 使用GPU加速
"""

import os
import sys
import json
import csv
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# 导入特征提取器
try:
    from feature_extractor import FeatureExtractor
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="SKU特征建库 - ViT-S16 DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接模式（从 sku_output/ 读取原始图片）
  python build_library.py --input ./sku_output --output ../sku_library
  
  # 增强模式（从 sku_augmentation.py 的输出读取）
  python build_library.py --input ./sku_library --output ../sku_library --use-aug-csv
  
  # 使用GPU加速
  python build_library.py --device cuda
        """
    )
    
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='./sku_output',
        help='输入目录 (默认: ./sku_output)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='../sku_library',
        help='输出目录 (默认: ../sku_library)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='推理设备 (cpu/cuda, 默认: cpu)'
    )
    
    parser.add_argument(
        '--use-aug-csv',
        action='store_true',
        help='使用 sku_augmentation.py 生成的 sku_library.csv 作为输入'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='微调模型路径 (默认: 使用预训练模型)'
    )
    
    return parser.parse_args()


def read_aug_csv(csv_path: Path, input_dir: Path) -> List[Dict]:
    """从 sku_augmentation.py 生成的 csv 读取数据"""
    csv_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 构建完整路径
            image_name = row['image_name']
            image_path = input_dir / image_name
            if image_path.exists():
                csv_rows.append({
                    "image_name": image_name,
                    "sku_id": row['sku_id'],
                    "label": row['label'],
                    "sku_name": row['sku_name'],
                    "path": str(image_path)
                })
    return csv_rows


def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir.resolve()}")
        return 1
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  SKU特征建库")
    print("=" * 70)
    print(f"  输入目录: {input_dir.resolve()}")
    print(f"  输出目录: {output_dir.resolve()}")
    print(f"  推理设备: {args.device}")
    print(f"  使用增强CSV: {args.use_aug_csv}")
    if args.model_path:
        print(f"  微调模型: {Path(args.model_path).resolve()}")
    else:
        print(f"  微调模型: (未使用，使用预训练模型)")
    print()
    
    # 1. 收集图片数据
    print("[1/5] 收集SKU图片...")
    csv_rows = []
    
    if args.use_aug_csv:
        # 增强模式：从 sku_library.csv 读取
        csv_path = input_dir / 'sku_library.csv'
        if not csv_path.exists():
            print(f"  错误: 找不到 sku_library.csv: {csv_path.resolve()}")
            return 1
        csv_rows = read_aug_csv(csv_path, input_dir)
    else:
        # 直接模式：从 sku_database.json 读取
        db_path = input_dir / 'sku_database.json'
        if not db_path.exists():
            print(f"  错误: 数据库文件不存在: {db_path.resolve()}")
            return 1
        
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except Exception as e:
            print(f"  错误: 数据库加载失败: {e}")
            return 1
        
        if "skus" in db:
            # 新结构: {"skus": [...]}
            for sku in db["skus"]:
                sku_id = sku.get("sku_id", "")
                sku_name = sku.get("sku_name", sku_id)
                members = sku.get("members", [])
                
                sku_dir = input_dir / sku_id
                if not sku_dir.exists():
                    continue
                
                for img_file in sku_dir.iterdir():
                    if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                        csv_rows.append({
                            "image_name": f"{sku_id}/{img_file.name}",
                            "sku_id": sku_id,
                            "label": sku_id,
                            "sku_name": sku_name,
                            "path": str(img_file)
                        })
        else:
            # 旧结构: {"sku_id": {...}}
            for sku_id, sku_info in db.items():
                sku_name = sku_info.get("name", sku_id)
                sku_dir = input_dir / sku_id
                if not sku_dir.exists():
                    continue
                
                for img_file in sku_dir.iterdir():
                    if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                        csv_rows.append({
                            "image_name": f"{sku_id}/{img_file.name}",
                            "sku_id": sku_id,
                            "label": sku_id,
                            "sku_name": sku_name,
                            "path": str(img_file)
                        })
    
    if not csv_rows:
        print(f"  ✗ 没有找到任何SKU图片")
        return 1
    
    sku_count = len(set(row['sku_id'] for row in csv_rows))
    print(f"  ✓ 找到 {sku_count} 个SKU, {len(csv_rows)} 张图片")
    print()
    
    # 复制图片到输出目录
    print("[1.5/5] 复制图片到输出目录...")
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    updated_csv_rows = []
    for row in csv_rows:
        src_path = Path(row['path'])
        
        # 保持文件夹结构：images/{sku_id}/{filename}
        sku_dir = images_output_dir / row['sku_id']
        sku_dir.mkdir(exist_ok=True)
        dst_path = sku_dir / src_path.name
        
        try:
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"  [!] 复制失败: {src_path.name}")
        
        # 更新路径到新位置供特征提取使用
        new_image_name = f"{row['sku_id']}/{src_path.name}"
        updated_row = row.copy()
        updated_row['image_name'] = new_image_name
        updated_row['path'] = str(dst_path)
        updated_csv_rows.append(updated_row)
    csv_rows = updated_csv_rows
    
    # 2. 初始化特征提取器
    print("[2/5] 初始化特征提取器...")
    try:
        extractor = FeatureExtractor(model_path=args.model_path, device=args.device)
        print(f"  ✓ ViT-S16 DINO 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return 1
    print()
    
    # 3. 提取特征（使用批量处理优化）
    print("[3/5] 提取特征...")
    
    # 预加载所有图片到内存
    print("  预加载图片...")
    images = []
    valid_indices = []
    for i, row in enumerate(csv_rows):
        img_path = Path(row['path'])
        try:
            images.append(Image.open(img_path))
            valid_indices.append(i)
        except Exception as e:
            print(f"  [!] 跳过 {row['image_name']}: {e}")
    
    print(f"  有效图片: {len(images)} 张")
    
    # 使用批量提取（CPU建议 batch_size=8）
    batch_size = 8 if args.device == 'cpu' else 32
    print(f"  批量处理中，批次大小: {batch_size}...")
    
    features_matrix = extractor.extract_batch(images, batch_size=batch_size)
    print(f"  ✓ 特征矩阵: {features_matrix.shape}")
    print()
    
    # 4. 保存输出
    print("[4/5] 保存输出文件...")
    
    # 保存特征矩阵
    features_out_path = output_dir / 'sku_features.npy'
    np.save(features_out_path, features_matrix)
    print(f"  ✓ 特征矩阵: {features_out_path.resolve()}")
    
    # 保存索引文件（保持 sku_augmentation.py 的格式）
    index_out_path = output_dir / 'sku_library.csv'
    with open(index_out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'sku_id', 'label', 'sku_name'])
        writer.writeheader()
        # 只保存需要的字段，image_name只用图片本身的名字
        for row in csv_rows:
            writer.writerow({
                "image_name": Path(row['image_name']).name,
                "sku_id": row["sku_id"],
                "label": row["label"],
                "sku_name": row["sku_name"]
            })
    print(f"  ✓ 索引文件: {index_out_path.resolve()}")
    
    print()
    print("=" * 70)
    print("  建库完成！")
    print("=" * 70)
    print()
    print(f"  统计:")
    print(f"    - SKU数量: {sku_count}")
    print(f"    - 图片数量: {len(csv_rows)}")
    print(f"    - 特征维度: {features_matrix.shape[1]}")
    print()
    print(f"  Web后端配置 sku_dir 为: {output_dir.resolve()}")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
