"""
随机裁剪拼接数据集生成脚本 V5 (可复现版 + 真实感优化)

核心功能：
1. 随机种子：支持设置随机种子，确保实验结果完全可复现
2. 策略模式：随机选择处理策略，避免多重处理叠加导致失真
3. 智能拼接：支持遮挡拼接和扩展拼接
4. 透视变换：优化了变换强度，避免严重畸变

使用方法：
    python generate_crop_stitch_v5.py --input pack_original --output mypack_crop --num-images 500 --seed 42
"""

import os
import cv2
import random
import argparse
import numpy as np
from glob import glob
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='随机裁剪拼接数据集生成脚本 V5')
    
    parser.add_argument('--input', required=True, help='输入数据集路径')
    parser.add_argument('--output', required=True, help='输出数据集路径')
    
    parser.add_argument('--num-images', type=int, default=1000,
                       help='生成的图像数量 (默认: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42，设为-1则不固定种子)')
    
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='数据集划分 (默认: train valid test)')
    parser.add_argument('--split-ratios', nargs='+', type=float, default=[0.7, 0.2, 0.1],
                       help='划分比例 (默认: 0.7 0.2 0.1)')
    
    parser.add_argument('--max-occlusion', type=float, default=0.40,
                       help='最大遮挡比例 (默认: 0.40)')
    parser.add_argument('--max-expand', type=float, default=0.30,
                       help='最大扩充比例 (默认: 0.30)')
    
    parser.add_argument('--min-density', type=int, default=3,
                       help='原图最少目标数 (默认: 3)')
    parser.add_argument('--density-weight', type=float, default=0.5,
                       help='高密度图片选择权重 (默认: 0.5)')
    
    return parser.parse_args()


def read_yolo_label(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls, x_center, y_center, width, height = map(float, parts[:5])
            
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            boxes.append([int(cls), x1, y1, x2, y2])
    
    return boxes


def apply_perspective_transform(image, boxes, angle_range=25):
    """
    透视变换：降低默认强度，避免严重失真
    """
    h, w = image.shape[:2]
    
    # 随机选择一种透视类型
    transform_type = random.choice(['top_narrow', 'bottom_narrow', 'left_narrow', 'right_narrow', 'general'])
    
    src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    max_shift = min(w, h) * (angle_range / 100) 
    
    if transform_type == 'top_narrow':
        shift = random.uniform(0.1, 0.3) * w
        dst_pts = np.array([[shift, 0], [w - shift, 0], [0, h], [w, h]], dtype=np.float32)
    elif transform_type == 'bottom_narrow':
        shift = random.uniform(0.1, 0.3) * w
        dst_pts = np.array([[0, 0], [w, 0], [shift, h], [w - shift, h]], dtype=np.float32)
    elif transform_type == 'left_narrow':
        shift = random.uniform(0.1, 0.3) * h
        dst_pts = np.array([[0, shift], [w, 0], [0, h - shift], [w, h]], dtype=np.float32)
    elif transform_type == 'right_narrow':
        shift = random.uniform(0.1, 0.3) * h
        dst_pts = np.array([[0, 0], [w, shift], [0, h], [w, h - shift]], dtype=np.float32)
    else:
        # 通用透视
        shifts = np.random.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32)
        dst_pts = src_pts + shifts

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 使用 BORDER_REPLICATE 填充边缘，比黑色填充更自然
    transformed = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    new_boxes = []
    for box in boxes:
        cls, x1, y1, x2, y2 = box
        corners = np.float32([[[x1, y1]], [[x2, y1]], [[x1, y2]], [[x2, y2]]])
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        xs = [transformed_corners[i][0][0] for i in range(4)]
        ys = [transformed_corners[i][0][1] for i in range(4)]
        
        new_x1 = max(0, min(xs))
        new_y1 = max(0, min(ys))
        new_x2 = min(w, max(xs))
        new_y2 = min(h, max(ys))
        
        # 过滤掉变形后过小的框
        if new_x2 > new_x1 + 5 and new_y2 > new_y1 + 5:
            new_boxes.append([cls, int(new_x1), int(new_y1), int(new_x2), int(new_y2)])
            
    return transformed, new_boxes, M


def apply_random_brightness(image, num_regions=1):
    """
    随机区域亮度调整 (优化：默认只调整1个区域，避免马赛克感)
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    for _ in range(num_regions):
        region_w = random.randint(w // 6, w // 3)
        region_h = random.randint(h // 6, h // 3)
        x = random.randint(0, w - region_w)
        y = random.randint(0, h - region_h)
        
        # 缩小亮度变化范围，避免过曝或过暗
        brightness_factor = random.uniform(0.85, 1.15)
        
        result[y:y+region_h, x:x+region_w] = np.clip(
            result[y:y+region_h, x:x+region_w] * brightness_factor,
            0, 255
        ).astype(np.uint8)
        
    return result


def apply_noise(image, noise_type='gaussian'):
    """
    噪声增强
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    if noise_type == 'gaussian':
        sigma = random.uniform(3, 10)  # 降低默认噪声强度
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        prob = random.uniform(0.001, 0.003)
        mask = np.random.choice([0, 1, 2], size=(h, w), p=[prob, prob, 1-2*prob])
        result[mask == 0] = 0
        result[mask == 1] = 255
    elif noise_type == 'speckle':
        var = random.uniform(0.005, 0.02)
        noise = np.random.randn(h, w, 3) * var * 255
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return result


def create_feathered_mask(h, w, feather_amount=10, direction=None):
    mask = np.ones((h, w), dtype=np.float32)
    
    if direction is None:
        for i in range(min(feather_amount, h // 2, w // 2)):
            alpha = i / feather_amount
            mask[i, :] = np.minimum(mask[i, :], alpha)
            mask[h - i - 1, :] = np.minimum(mask[h - i - 1, :], alpha)
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, w - i - 1] = np.minimum(mask[:, w - i - 1], alpha)
    elif direction == 'left':
        for i in range(min(feather_amount, w)):
            alpha = i / feather_amount
            mask[:, i] = alpha
    elif direction == 'right':
        for i in range(min(feather_amount, w)):
            alpha = i / feather_amount
            mask[:, w - i - 1] = alpha
    elif direction == 'top':
        for i in range(min(feather_amount, h)):
            alpha = i / feather_amount
            mask[i, :] = alpha
    elif direction == 'bottom':
        for i in range(min(feather_amount, h)):
            alpha = i / feather_amount
            mask[h - i - 1, :] = alpha
            
    return mask


def adjust_brightness(src, target_brightness):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_mean = np.mean(src_gray)
    if src_mean > 0:
        ratio = target_brightness / src_mean
        ratio = np.clip(ratio, 0.8, 1.2)
        return cv2.convertScaleAbs(src, alpha=ratio, beta=0)
    return src


def load_dataset(input_dir, splits, min_density=0):
    logger.info("加载数据集...")
    data = []
    
    for split in splits:
        image_dir = os.path.join(input_dir, split, "images")
        label_dir = os.path.join(input_dir, split, "labels")
        
        image_paths = glob(os.path.join(image_dir, "*.jpg"))
        image_paths.extend(glob(os.path.join(image_dir, "*.png")))
        
        for image_path in image_paths:
            name = os.path.basename(image_path).rsplit(".", 1)[0]
            label_path = os.path.join(label_dir, name + ".txt")
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            boxes = read_yolo_label(label_path, img_width, img_height)
            
            if min_density > 0 and len(boxes) < min_density:
                continue
            
            data.append({
                'image': image,
                'boxes': boxes,
                'width': img_width,
                'height': img_height,
                'name': name,
                'density': len(boxes)
            })
    
    data.sort(key=lambda x: x['density'], reverse=True)
    
    logger.info(f"加载完成: {len(data)} 张图片")
    if data:
        densities = [d['density'] for d in data]
        logger.info(f"密度统计: 最小={min(densities)}, 最大={max(densities)}, 平均={np.mean(densities):.1f}")
    
    return data


def select_high_density_image(data, weight=0.7):
    if not data:
        return None
    
    if random.random() < weight:
        top_count = max(1, len(data) // 3)
        idx = random.randint(0, top_count - 1)
    else:
        idx = random.randint(0, len(data) - 1)
    
    return idx


def horizontal_stitch(base_img, base_boxes, crop_img, crop_boxes, max_occlusion, max_expand):
    base_h, base_w = base_img.shape[:2]
    crop_h, crop_w = crop_img.shape[:2]
    
    scale = base_h / crop_h
    new_crop_w = int(crop_w * scale)
    crop_img_resized = cv2.resize(crop_img, (new_crop_w, base_h))
    
    scaled_crop_boxes = []
    for box in crop_boxes:
        cls, cx1, cy1, cx2, cy2 = box
        scaled_crop_boxes.append([cls, int(cx1 * scale), int(cy1 * scale), int(cx2 * scale), int(cy2 * scale)])
    
    mode = random.choice(['occlusion', 'expand'])
    
    if mode == 'occlusion':
        max_occlude_width = int(base_w * max_occlusion)
        occlude_width = random.randint(int(max_occlude_width * 0.3), max_occlude_width)
        
        if new_crop_w > occlude_width:
            start_x = random.randint(0, new_crop_w - occlude_width)
            crop_img_resized = crop_img_resized[:, start_x:start_x + occlude_width]
            new_crop_w = occlude_width
            
            adjusted_boxes = []
            for box in scaled_crop_boxes:
                cls, cx1, cy1, cx2, cy2 = box
                nx1 = cx1 - start_x
                nx2 = cx2 - start_x
                
                if nx2 > 0 and nx1 < occlude_width:
                    nx1 = max(0, nx1)
                    nx2 = min(occlude_width, nx2)
                    if nx2 > nx1 + 5:
                        adjusted_boxes.append([cls, nx1, cy1, nx2, cy2])
            
            scaled_crop_boxes = adjusted_boxes
        
        mask = create_feathered_mask(base_h, new_crop_w, feather_amount=15, direction='left')
        target_brightness = np.mean(base_img[:, -occlude_width:])
        crop_img_resized = adjust_brightness(crop_img_resized, target_brightness)
        
        result = base_img.copy()
        x_start = base_w - new_crop_w
        
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        result[:, x_start:] = (crop_img_resized * mask_3d + result[:, x_start:] * (1 - mask_3d)).astype(np.uint8)
        
        new_boxes = []
        for box in base_boxes:
            cls, x1, y1, x2, y2 = box
            if x2 > x_start:
                x2 = min(x2, x_start + new_crop_w)
                if x2 > x1 and (x2 - x1) > 10:
                    new_boxes.append([cls, x1, y1, x2, y2])
            else:
                new_boxes.append(box)
        
        for box in scaled_crop_boxes:
            cls, cx1, cy1, cx2, cy2 = box
            new_x1 = x_start + cx1
            new_x2 = x_start + cx2
            
            if new_x2 > x_start and new_x1 < base_w:
                new_x1 = max(new_x1, x_start)
                new_x2 = min(new_x2, base_w)
                if new_x2 > new_x1 + 10:
                    new_boxes.append([cls, new_x1, cy1, new_x2, cy2])
        
        return result, new_boxes, base_w, base_h
        
    else:
        max_expand_width = int(base_w * max_expand)
        expand_width = min(new_crop_w, max_expand_width)
        
        if new_crop_w > expand_width:
            start_x = random.randint(0, new_crop_w - expand_width)
            crop_img_resized = crop_img_resized[:, start_x:start_x + expand_width]
            new_crop_w = expand_width
            
            adjusted_boxes = []
            for box in scaled_crop_boxes:
                cls, cx1, cy1, cx2, cy2 = box
                nx1 = cx1 - start_x
                nx2 = cx2 - start_x
                if nx2 > nx1 + 5:
                    adjusted_boxes.append([cls, nx1, cy1, nx2, cy2])
            scaled_crop_boxes = adjusted_boxes
        
        mask = create_feathered_mask(base_h, new_crop_w, feather_amount=15, direction='left')
        target_brightness = np.mean(base_img[:, -50:] if base_w > 50 else base_img)
        crop_img_resized = adjust_brightness(crop_img_resized, target_brightness)
        
        new_width = base_w + new_crop_w
        result = np.zeros((base_h, new_width, 3), dtype=np.uint8)
        result[:, :base_w] = base_img
        
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        result[:, base_w:] = (crop_img_resized * mask_3d + result[:, base_w:] * (1 - mask_3d)).astype(np.uint8)
        
        new_boxes = []
        for box in base_boxes:
            new_boxes.append(box)
        
        for box in scaled_crop_boxes:
            cls, cx1, cy1, cx2, cy2 = box
            new_x1 = base_w + cx1
            new_x2 = base_w + cx2
            if new_x2 > new_x1 + 10:
                new_boxes.append([cls, new_x1, cy1, new_x2, cy2])
        
        return result, new_boxes, new_width, base_h


def vertical_stitch(base_img, base_boxes, crop_img, crop_boxes, max_occlusion, max_expand):
    base_h, base_w = base_img.shape[:2]
    crop_h, crop_w = crop_img.shape[:2]
    
    scale = base_w / crop_w
    new_crop_h = int(crop_h * scale)
    crop_img_resized = cv2.resize(crop_img, (base_w, new_crop_h))
    
    scaled_crop_boxes = []
    for box in crop_boxes:
        cls, cx1, cy1, cx2, cy2 = box
        scaled_crop_boxes.append([cls, int(cx1 * scale), int(cy1 * scale), int(cx2 * scale), int(cy2 * scale)])
    
    mode = random.choice(['occlusion', 'expand'])
    
    if mode == 'occlusion':
        max_occlude_height = int(base_h * max_occlusion)
        occlude_height = random.randint(int(max_occlude_height * 0.3), max_occlude_height)
        
        if new_crop_h > occlude_height:
            start_y = random.randint(0, new_crop_h - occlude_height)
            crop_img_resized = crop_img_resized[start_y:start_y + occlude_height, :]
            new_crop_h = occlude_height
            
            adjusted_boxes = []
            for box in scaled_crop_boxes:
                cls, cx1, cy1, cx2, cy2 = box
                ny1 = cy1 - start_y
                ny2 = cy2 - start_y
                
                if ny2 > 0 and ny1 < occlude_height:
                    ny1 = max(0, ny1)
                    ny2 = min(occlude_height, ny2)
                    if ny2 > ny1 + 5:
                        adjusted_boxes.append([cls, cx1, ny1, cx2, ny2])
            
            scaled_crop_boxes = adjusted_boxes
        
        mask = create_feathered_mask(new_crop_h, base_w, feather_amount=15, direction='top')
        target_brightness = np.mean(base_img[-occlude_height:, :])
        crop_img_resized = adjust_brightness(crop_img_resized, target_brightness)
        
        result = base_img.copy()
        y_start = base_h - new_crop_h
        
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        result[y_start:, :] = (crop_img_resized * mask_3d + result[y_start:, :] * (1 - mask_3d)).astype(np.uint8)
        
        new_boxes = []
        for box in base_boxes:
            cls, x1, y1, x2, y2 = box
            if y2 > y_start:
                y2 = min(y2, y_start + new_crop_h)
                if y2 > y1 and (y2 - y1) > 10:
                    new_boxes.append([cls, x1, y1, x2, y2])
            else:
                new_boxes.append(box)
        
        for box in scaled_crop_boxes:
            cls, cx1, cy1, cx2, cy2 = box
            new_y1 = y_start + cy1
            new_y2 = y_start + cy2
            
            if new_y2 > y_start and new_y1 < base_h:
                new_y1 = max(new_y1, y_start)
                new_y2 = min(new_y2, base_h)
                if new_y2 > new_y1 + 10:
                    new_boxes.append([cls, cx1, new_y1, cx2, new_y2])
        
        return result, new_boxes, base_w, base_h
        
    else:
        max_expand_height = int(base_h * max_expand)
        expand_height = min(new_crop_h, max_expand_height)
        
        if new_crop_h > expand_height:
            start_y = random.randint(0, new_crop_h - expand_height)
            crop_img_resized = crop_img_resized[start_y:start_y + expand_height, :]
            new_crop_h = expand_height
            
            adjusted_boxes = []
            for box in scaled_crop_boxes:
                cls, cx1, cy1, cx2, cy2 = box
                ny1 = cy1 - start_y
                ny2 = cy2 - start_y
                if ny2 > ny1 + 5:
                    adjusted_boxes.append([cls, cx1, ny1, cx2, ny2])
            scaled_crop_boxes = adjusted_boxes
        
        mask = create_feathered_mask(new_crop_h, base_w, feather_amount=15, direction='top')
        target_brightness = np.mean(base_img[-50:, :] if base_h > 50 else base_img)
        crop_img_resized = adjust_brightness(crop_img_resized, target_brightness)
        
        new_height = base_h + new_crop_h
        result = np.zeros((new_height, base_w, 3), dtype=np.uint8)
        result[:base_h, :] = base_img
        
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        result[base_h:, :] = (crop_img_resized * mask_3d + result[base_h:, :] * (1 - mask_3d)).astype(np.uint8)
        
        new_boxes = []
        for box in base_boxes:
            new_boxes.append(box)
        
        for box in scaled_crop_boxes:
            cls, cx1, cy1, cx2, cy2 = box
            new_y1 = base_h + cy1
            new_y2 = base_h + cy2
            if new_y2 > new_y1 + 10:
                new_boxes.append([cls, cx1, new_y1, cx2, new_y2])
        
        return result, new_boxes, base_w, new_height


def generate_image(data, args):
    if len(data) < 2:
        return None, []

    # 1. 选择图片
    idx1 = select_high_density_image(data, args.density_weight)
    idx2 = random.randint(0, len(data) - 1)
    while idx2 == idx1:
        idx2 = random.randint(0, len(data) - 1)

    base_data = data[idx1]
    crop_data = data[idx2]

    base_img = base_data['image'].copy()
    base_boxes = base_data['boxes'].copy()
    crop_img = crop_data['image'].copy()
    crop_boxes = crop_data['boxes'].copy()

    # ==========================================
    # 核心修改：随机选择一种处理策略
    # ==========================================
    mode = random.choice(['geometric', 'light', 'noise', 'clean'])
    
    # 默认全部关闭
    p_perspective = 0.0
    p_brightness = 0.0
    p_noise = 0.0
    perspective_strength = 20 # 透视强度默认较低

    if mode == 'geometric':
        # 几何模式：重点做透视，不做光照和噪声干扰，保持图像清晰但角度多变
        p_perspective = 1.0
        perspective_strength = 35 # 稍微加强一点透视
        
    elif mode == 'light':
        # 光照模式：不做透视，重点模拟光照变化
        p_perspective = 0.3 # 轻微透视
        p_brightness = 1.0
        
    elif mode == 'noise':
        # 噪声模式：模拟低画质，不做透视
        p_perspective = 0.0
        p_noise = 1.0
        
    elif mode == 'clean':
        # 纯净模式：只做拼接，没有任何额外干扰，最真实
        p_perspective = 0.0
        p_brightness = 0.0
        p_noise = 0.0

    # 2. 执行透视变换 (使用动态概率和强度)
    if random.random() < p_perspective:
        base_img, base_boxes, _ = apply_perspective_transform(base_img, base_boxes, angle_range=perspective_strength)
        
    if random.random() < p_perspective: # 粘贴图也做对应处理
        crop_img, crop_boxes, _ = apply_perspective_transform(crop_img, crop_boxes, angle_range=perspective_strength)

    # 3. 拼接 (核心操作，始终执行)
    direction = random.choice(['horizontal', 'vertical'])
    if direction == 'horizontal':
        result, new_boxes, new_w, new_h = horizontal_stitch(
            base_img, base_boxes, crop_img, crop_boxes, args.max_occlusion, args.max_expand
        )
    else:
        result, new_boxes, new_w, new_h = vertical_stitch(
            base_img, base_boxes, crop_img, crop_boxes, args.max_occlusion, args.max_expand
        )

    # 4. 后处理 (根据模式执行)
    # 随机区域亮度调整
    if random.random() < p_brightness:
        # 建议减少区域数量为1个，避免像马赛克
        result = apply_random_brightness(result, num_regions=1)

    # 添加噪声
    if random.random() < p_noise:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        result = apply_noise(result, noise_type)

    # 5. 生成标注
    annotations = []
    for box in new_boxes:
        cls, x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / new_w
        y_center = ((y1 + y2) / 2) / new_h
        width = (x2 - x1) / new_w
        height = (y2 - y1) / new_h
        if width > 0.01 and height > 0.01:
            annotations.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return result, annotations


def generate_dataset(args):
    # 设置随机种子
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"随机种子设置为: {args.seed}")
    else:
        logger.info("未固定随机种子，每次运行结果将不同")

    os.makedirs(args.output, exist_ok=True)
    for split in args.splits:
        os.makedirs(os.path.join(args.output, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output, split, "labels"), exist_ok=True)

    data = load_dataset(args.input, args.splits, args.min_density)
    
    if len(data) < 2:
        logger.error("数据集图片数量不足")
        return

    logger.info(f"开始生成 {args.num_images} 张图像...")
    
    stats = {'total_objs': 0, 'total_imgs': 0}
    
    for i in range(args.num_images):
        result, annotations = generate_image(data, args)
        
        if result is None:
            continue
        
        stats['total_objs'] += len(annotations)
        stats['total_imgs'] += 1
        
        r = random.random()
        if r < args.split_ratios[0]:
            split = args.splits[0]
        elif r < args.split_ratios[0] + args.split_ratios[1]:
            split = args.splits[1]
        else:
            split = args.splits[2]
            
        save_name = f"stitch_{i:05d}"
        cv2.imwrite(os.path.join(args.output, split, "images", f"{save_name}.jpg"), result)
        
        with open(os.path.join(args.output, split, "labels", f"{save_name}.txt"), 'w') as f:
            f.writelines(annotations)
        
        if (i + 1) % 100 == 0:
            logger.info(f"进度: {i+1}/{args.num_images}, 平均目标数: {stats['total_objs']/stats['total_imgs']:.1f}")

    src_yaml = os.path.join(args.input, "data.yaml")
    if os.path.exists(src_yaml):
        shutil.copy(src_yaml, args.output)

    logger.info(f"生成完成! 总图像: {stats['total_imgs']}, 总目标: {stats['total_objs']}")


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
