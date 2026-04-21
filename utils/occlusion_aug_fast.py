"""
遮挡增强数据集预处理 - 多进程加速版
使用方法：
    python occlusion_aug_fast.py --source /path/to/dataset.yaml --output /path/to/output --aug-ratio 0.3 --workers 8
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 禁用哈希随机化，确保可复现性
os.environ['PYTHONHASHSEED'] = '0'


def get_deterministic_hash(filename):
    """生成确定性哈希值"""
    import hashlib
    return int(hashlib.md5(filename.encode()).hexdigest(), 16) % 1000000


OCCLUSION_AUG_CONFIG = {
    'aug_prob': 0.5,           # 每个样本增强概率
    'target_classes': [1, 3],  # 目标类别（两个occlusion类别）
    'num_holes_range': (1, 3), # 挖洞数量
    'hole_size_range': (10, 30), # 洞大小（像素）
}


def parse_yolo_seg_label(label_path: str, img_h: int, img_w: int):
    """解析YOLO分割标注（多边形格式）"""
    if not os.path.exists(label_path):
        return [], [], []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    classes, polygons, masks = [], [], []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                points.append([x, y])
            
            polygon = np.array(points, dtype=np.float32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
            
            classes.append(cls)
            polygons.append(polygon)
            masks.append(mask)
        except (ValueError, IndexError):
            continue
    
    return classes, polygons, masks


def mask_to_polygon(mask: np.ndarray):
    """从mask提取多边形轮廓"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    return approx.reshape(-1, 2).astype(np.float32)


def apply_occlusion_augmentation(image_path, label_path, output_image_path, output_label_path, config=None):
    """对单张分割图像应用遮挡增强"""
    cfg = {**OCCLUSION_AUG_CONFIG, **(config or {})}
    
    if random.random() > cfg['aug_prob']:
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    H, W = image.shape[:2]
    classes, polygons, masks = parse_yolo_seg_label(label_path, H, W)
    
    if not classes:
        return False
    
    target_indices = [i for i, c in enumerate(classes) if c in cfg['target_classes']]
    if not target_indices:
        return False
    
    image = image.copy()
    augmented = False
    
    for idx in target_indices:
        mask = masks[idx]
        if mask.sum() == 0:
            continue
        
        kernel = np.ones((5, 5), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        boundary_coords = np.where(boundary > 0)
        
        if len(boundary_coords[0]) == 0:
            continue
        
        num_holes = random.randint(*cfg['num_holes_range'])
        for _ in range(num_holes):
            point_idx = random.randint(0, len(boundary_coords[0]) - 1)
            cy, cx = boundary_coords[0][point_idx], boundary_coords[1][point_idx]
            hole_size = random.randint(*cfg['hole_size_range'])
            
            y1 = max(0, cy - hole_size // 2)
            x1 = max(0, cx - hole_size // 2)
            y2 = min(H, y1 + hole_size)
            x2 = min(W, x1 + hole_size)
            
            fill_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            image[y1:y2, x1:x2] = fill_color
            masks[idx][y1:y2, x1:x2] = 0
        
        augmented = True
    
    if not augmented:
        return False
    
    with open(output_label_path, 'w') as f:
        for cls, mask in zip(classes, masks):
            if mask.sum() == 0:
                continue
            new_polygon = mask_to_polygon(mask)
            if new_polygon is None or len(new_polygon) < 3:
                continue
            
            coords = []
            for x, y in new_polygon:
                coords.extend([x / W, y / H])
            coords_str = ' '.join([f'{c:.6f}' for c in coords])
            f.write(f"{cls} {coords_str}\n")
    
    cv2.imwrite(output_image_path, image)
    return True


def process_single_image(args):
    """处理单张图像（用于多进程）- 替换式增强"""
    img_file, label_file, out_train_images, out_train_labels, aug_ratio, seed = args
    
    # 每个进程独立设置随机种子（使用确定性哈希）
    file_hash = get_deterministic_hash(img_file.name)
    random.seed(seed + file_hash)
    np.random.seed(seed + file_hash)
    
    # 决定是否增强
    if random.random() < aug_ratio:
        # 替换式：用增强版本覆盖原始文件名
        aug_img = out_train_images / img_file.name
        aug_label = out_train_labels / label_file.name
        try:
            if apply_occlusion_augmentation(str(img_file), str(label_file), str(aug_img), str(aug_label)):
                return 1
        except Exception:
            # 增强失败，回退到复制原始
            pass
    
    # 无论增强是否成功，都复制原始文件（确保数据不丢失）
    shutil.copy(img_file, out_train_images / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, out_train_labels / label_file.name)
    
    return 0


def preprocess_dataset_with_occlusion_fast(source_yaml, output_dir, aug_ratio=0.3, num_workers=8, seed=0):
    """预处理数据集：多进程并行生成遮挡增强版本"""
    random.seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(source_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    source_path = Path(data_config['path'])
    
    # 创建目录
    out_train_images = output_dir / 'images' / 'train'
    out_train_labels = output_dir / 'labels' / 'train'
    out_val_images = output_dir / 'images' / 'val'
    out_val_labels = output_dir / 'labels' / 'val'
    
    for d in [out_train_images, out_train_labels, out_val_images, out_val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 收集训练集文件（排序确保顺序一致）
    train_images_dir = source_path / 'images' / 'train'
    train_labels_dir = source_path / 'labels' / 'train'
    image_files = sorted(list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png')))
    
    print("=" * 60)
    print("开始预处理数据集（遮挡增强 - 多进程加速 - 替换式）")
    print(f"原始数据集: {source_yaml}")
    print(f"训练集图像数: {len(image_files)}")
    print(f"增强比例: {aug_ratio}")
    print(f"并行进程数: {num_workers}")
    print(f"增强模式: 替换式（训练集大小不变，公平对比）")
    print("=" * 60)
    
    # 准备任务列表
    tasks = []
    for img_file in image_files:
        label_file = train_labels_dir / (img_file.stem + '.txt')
        tasks.append((img_file, label_file, out_train_images, out_train_labels, aug_ratio, seed))
    
    # 多进程处理
    aug_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="预处理"))
        aug_count = sum(results)
    
    # 复制验证集
    if 'val' in data_config:
        val_images_dir = source_path / 'images' / 'val'
        val_labels_dir = source_path / 'labels' / 'val'
        if val_images_dir.exists():
            val_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
            for img_file in tqdm(val_files, desc="复制验证集"):
                shutil.copy(img_file, out_val_images / img_file.name)
                label_file = val_labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    shutil.copy(label_file, out_val_labels / label_file.name)
    
    # 生成配置
    output_config = data_config.copy()
    output_config['path'] = str(output_dir.absolute())
    output_config['train'] = 'images/train'
    output_config['val'] = 'images/val'
    
    output_yaml = output_dir / 'dataset.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(output_config, f)
    
    print(f"\n预处理完成！增强样本数: {aug_count}")
    print(f"输出配置: {output_yaml}")
    
    return str(output_yaml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='遮挡增强数据集预处理 - 多进程加速版')
    parser.add_argument('--source', type=str, required=True, help='原始数据集yaml路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--aug-ratio', type=float, default=0.3, help='增强比例（默认0.3）')
    parser.add_argument('--workers', type=int, default=8, help='并行进程数（默认8）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子（默认0）')
    
    args = parser.parse_args()
    
    preprocess_dataset_with_occlusion_fast(
        source_yaml=args.source,
        output_dir=args.output,
        aug_ratio=args.aug_ratio,
        num_workers=args.workers,
        seed=args.seed
    )
