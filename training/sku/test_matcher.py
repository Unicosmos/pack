"""
SKU匹配调试脚本

使用方法：
    python test_matcher.py --image path/to/test.jpg
    python test_matcher.py --image path/to/test.jpg --sku-dir ../sku_library --top-k 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "web" / "backend" / "core"))
from matcher import SKUMatcher


def main():
    parser = argparse.ArgumentParser(description="SKU匹配调试脚本")
    parser.add_argument("--image", "-i", type=str, required=True, help="测试图片路径")
    parser.add_argument("--sku-dir", "-s", type=str, default="../sku_library", help="SKU库目录")
    parser.add_argument("--yolo", "-y", type=str, default=None, help="YOLO模型路径（不需要可留空）")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="返回Top-K候选")
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="相似度阈值")
    parser.add_argument("--ratio", "-r", type=float, default=1.2, help="Ratio Test阈值")
    args = parser.parse_args()

    test_image_path = Path(args.image)
    if not test_image_path.exists():
        print(f"错误: 测试图片不存在: {test_image_path}")
        return 1

    print("=" * 70)
    print("  SKU匹配调试工具")
    print("=" * 70)
    print(f"  测试图片: {test_image_path}")
    print(f"  SKU库目录: {args.sku_dir}")
    print(f"  Top-K: {args.top_k}")
    print(f"  相似度阈值: {args.threshold}")
    print(f"  Ratio阈值: {args.ratio}")
    print()

    print("[1/3] 初始化SKUMatcher...")
    try:
        matcher = SKUMatcher(
            model_path=args.yolo or "",
            sku_dir=args.sku_dir,
            feature_dim=384,
            match_threshold=args.threshold,
            ratio_threshold=args.ratio,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"  初始化失败: {e}")
        return 1

    if not matcher.is_ready():
        print("  错误: SKUMatcher未就绪，请检查特征库是否存在")
        return 1

    print(f"  ✓ SKUMatcher已就绪")
    print(f"  ✓ 特征矩阵: {matcher.sku_features.shape}")
    print(f"  ✓ 索引数量: {len(matcher.sku_labels)}")
    print()

    print("[2/3] 加载测试图片...")
    try:
        image = Image.open(test_image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        print(f"  ✓ 图片加载成功: {image.size}")
    except Exception as e:
        print(f"  加载失败: {e}")
        return 1
    print()

    print("[3/3] 执行匹配...")
    try:
        query_feat = matcher.extract_feature(image)
        print(f"  ✓ 特征提取成功")
        print(f"    特征维度: {query_feat.shape}")
        print(f"    特征范数: {np.linalg.norm(query_feat):.4f}")
    except Exception as e:
        print(f"  特征提取失败: {e}")
        return 1

    result = matcher.match_sku(query_feat, threshold=args.threshold, ratio_threshold=args.ratio)

    print()
    print("=" * 70)
    print("  匹配结果")
    print("=" * 70)
    print(f"  最佳SKU ID: {result.sku_id}")
    print(f"  相似度: {result.similarity:.4f}")
    print(f"  Ratio: {result.ratio}")
    print(f"  状态: {result.status}")
    print()
    print("  Top-K 候选:")
    print("-" * 50)
    for i, label_info in enumerate(result.top5_labels):
        print(f"    {i+1}. {label_info['label']:<30} {label_info['similarity']:.4f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
