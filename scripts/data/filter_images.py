#!/usr/bin/env python
"""
图片筛选工具
过滤掉特征不明显和长宽比异常的图片

使用方法:
    python filter_images.py --input /path/to/images --output /path/to/output
    python filter_images.py -i /path/to/images -o /path/to/output
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

def calculate_aspect_ratio(image: Image.Image) -> float:
    """计算图片宽高比"""
    width, height = image.size
    if height == 0:
        return float('inf')
    return width / height

def calculate_entropy(image: Image.Image) -> float:
    """计算图片熵值（衡量信息量/特征丰富度）"""
    if image.mode != 'L':
        image = image.convert('L')
    
    histogram = image.histogram()
    total_pixels = sum(histogram)
    
    entropy = 0.0
    for count in histogram:
        if count > 0:
            probability = count / total_pixels
            entropy -= probability * (probability ** 0.5)
    
    return entropy

def is_blurry(image: Image.Image, threshold: float = 50.0) -> bool:
    """检测图片是否模糊（使用拉普拉斯方差）"""
    import numpy as np
    from scipy.ndimage import laplace
    
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    laplacian = laplace(img_array)
    variance = laplacian.var()
    
    return variance < threshold

def is_blank_or_uniform(image: Image.Image, threshold: float = 0.03) -> bool:
    """检测图片是否几乎为空白或纯色（无法识别任何商品）"""
    import numpy as np
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # 计算每个通道的标准差
    std_r = np.std(img_array[:, :, 0])
    std_g = np.std(img_array[:, :, 1])
    std_b = np.std(img_array[:, :, 2])
    
    avg_std = (std_r + std_g + std_b) / 3
    
    return avg_std < (255 * threshold)

def is_low_contrast(image: Image.Image, threshold: float = 0.1) -> bool:
    """检测图片是否对比度低"""
    if image.mode != 'L':
        image = image.convert('L')
    
    histogram = image.histogram()
    total = sum(histogram)
    
    mean = sum(i * count for i, count in enumerate(histogram)) / total
    variance = sum(count * (i - mean) ** 2 for i, count in enumerate(histogram)) / total
    std = variance ** 0.5
    normalized_std = std / 255.0
    
    return normalized_std < threshold

def filter_image(image_path: Path, min_aspect_ratio: float = 0.15, 
                max_aspect_ratio: float = 7.0, blur_threshold: float = 50.0) -> tuple:
    """
    判断图片是否应该被保留（只过滤人眼无法识别商品的极端情况）
    
    Args:
        image_path: 图片路径
        min_aspect_ratio: 最小宽高比
        max_aspect_ratio: 最大宽高比
        blur_threshold: 模糊阈值（拉普拉斯方差，数值越低越严格）
    
    Returns:
        (保留?, 过滤原因, 详细信息)
        保留? - True表示保留，False表示过滤
        过滤原因 - None表示保留，否则为过滤原因字符串
        详细信息 - 附加信息（如宽高比等）
    """
    try:
        with Image.open(image_path) as img:
            # 检查长宽比
            aspect_ratio = calculate_aspect_ratio(img)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                reason = "长宽比异常"
                detail = f"宽高比: {aspect_ratio:.2f} (范围: [{min_aspect_ratio}, {max_aspect_ratio}])"
                print(f"  ❌ [{reason}] {image_path.name} - {detail}")
                return False, reason, detail
            
            # 检查是否几乎为空白或纯色（如纸箱空白面）
            if is_blank_or_uniform(img):
                reason = "特征缺失"
                detail = "图片几乎为空白或纯色，无法识别商品"
                print(f"  ❌ [{reason}] {image_path.name} - {detail}")
                return False, reason, detail
            
            # 检查严重模糊（只有非常模糊的图片才过滤）
            if is_blurry(img, blur_threshold):
                reason = "严重模糊"
                detail = "图片过于模糊，无法识别商品"
                print(f"  ❌ [{reason}] {image_path.name}")
                return False, reason, detail
            
            print(f"  ✅ [保留] {image_path.name}")
            return True, None, None
            
    except Exception as e:
        reason = "读取失败"
        detail = str(e)
        print(f"  ❌ [{reason}] {image_path.name} - {detail}")
        return False, reason, detail

def main():
    parser = argparse.ArgumentParser(
        description='图片筛选工具 - 过滤特征不明显和长宽比异常的图片',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入图片目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--min-aspect',
        type=float,
        default=0.15,
        help='最小宽高比 (默认: 0.15)'
    )
    
    parser.add_argument(
        '--max-aspect',
        type=float,
        default=7.0,
        help='最大宽高比 (默认: 7.0)'
    )
    
    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=50.0,
        help='模糊检测阈值（数值越低越严格，默认: 50.0）'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='启用严格模式，过滤更多低质量图片'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成过滤报告文件'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在 - {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    # 递归遍历所有子目录
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f'*{ext}'))
        image_paths.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    print(f"=" * 70)
    print(f"图片筛选工具")
    print(f"=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到图片: {len(image_paths)} 张")
    # 严格模式参数
    blur_thresh = args.blur_threshold
    if args.strict:
        blur_thresh = 80.0  # 更宽松的模糊阈值，会过滤更多模糊图片
    
    print(f"筛选参数:")
    print(f"  - 宽高比范围: [{args.min_aspect}, {args.max_aspect}]")
    print(f"  - 模糊阈值: {blur_thresh}")
    print(f"  - 模式: {'严格模式' if args.strict else '标准模式'}")
    print("-" * 70)
    
    kept_count = 0
    filtered_count = 0
    filtered_details = []
    reason_stats = defaultdict(int)
    
    for img_path in image_paths:
        keep, reason, detail = filter_image(img_path, args.min_aspect, args.max_aspect, blur_thresh)
        
        if keep:
            # 保持原有的目录结构
            rel_path = img_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(img_path) as img:
                img.save(output_path, quality=95)
            kept_count += 1
        else:
            filtered_count += 1
            reason_stats[reason] += 1
            filtered_details.append({
                'filename': str(img_path.relative_to(input_dir)),
                'reason': reason,
                'detail': detail
            })
    
    print("-" * 70)
    print(f"筛选完成！")
    print(f"保留图片: {kept_count} 张")
    print(f"过滤图片: {filtered_count} 张")
    print("-" * 70)
    
    if filtered_count > 0:
        print(f"\n过滤原因统计:")
        for reason, count in sorted(reason_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / filtered_count) * 100
            print(f"  • {reason}: {count} 张 ({percentage:.1f}%)")
        
        if args.report:
            report_path = output_dir / "filter_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("图片筛选报告\n")
                f.write("=" * 70 + "\n")
                f.write(f"输入目录: {input_dir}\n")
                f.write(f"输出目录: {output_dir}\n")
                f.write(f"筛选时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 70 + "\n")
                f.write(f"总图片数: {len(image_paths)}\n")
                f.write(f"保留图片: {kept_count}\n")
                f.write(f"过滤图片: {filtered_count}\n")
                f.write("-" * 70 + "\n")
                f.write("过滤原因统计:\n")
                for reason, count in sorted(reason_stats.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  • {reason}: {count} 张\n")
                f.write("-" * 70 + "\n")
                f.write("过滤详情:\n")
                for item in filtered_details:
                    f.write(f"  • {item['filename']}: {item['reason']}")
                    if item['detail']:
                        f.write(f" - {item['detail']}")
                    f.write("\n")
            print(f"\n📋 过滤报告已保存到: {report_path}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()