"""
特征聚类模块
从箱体特征自动生成SKU库

使用方法:
    python sku_clustering.py --input features.pkl --output-dir ./sku_output
    
    # 调整聚类参数
    python sku_clustering.py --input features.pkl --eps 0.4 --min-samples 3
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_features_from_results(results: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    从检测结果提取特征矩阵
    
    Args:
        results: 特征提取模块的输出结果
        
    Returns:
        features: 特征矩阵 (N, 2048)
        metadata: 每个特征对应的元信息
    """
    features = []
    metadata = []
    
    for result in results:
        image_path = result.get("image_path", "")
        
        if "feature_vector" in result and result["feature_vector"] is not None:
            fv = result["feature_vector"]
            features.append(fv)
            metadata.append({
                "image_path": image_path,
                "bbox": None,
                "confidence": None,
            })
        
        if "detections" in result:
            for det in result["detections"]:
                if "feature_vector" in det and det["feature_vector"] is not None:
                    fv = det["feature_vector"]
                    features.append(fv)
                    metadata.append({
                        "image_path": image_path,
                        "bbox": det.get("bbox"),
                        "confidence": det.get("confidence"),
                    })
    
    features = np.array(features, dtype=np.float32)
    
    print(f"总箱体数: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    
    return features, metadata


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    L2归一化特征向量
    
    Args:
        features: 特征矩阵 (N, D)
        
    Returns:
        归一化后的特征矩阵
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = features / norms
    
    print(f"特征已L2归一化")
    
    return normalized


def reduce_dimensions(
    features: np.ndarray, 
    n_components: int = 128,
    variance_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA]:
    """
    PCA降维
    
    Args:
        features: 特征矩阵 (N, D)
        n_components: 目标维度
        variance_threshold: 方差保留阈值
        
    Returns:
        reduced_features: 降维后的特征
        pca: PCA模型
    """
    pca = PCA(n_components=min(n_components, features.shape[1], features.shape[0]))
    reduced_features = pca.fit_transform(features)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    
    print(f"PCA降维: {features.shape[1]} -> {reduced_features.shape[1]}")
    print(f"保留方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    return reduced_features, pca


def cluster_features(
    features: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 2,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    DBSCAN聚类
    
    Args:
        features: 特征矩阵
        eps: 邻域半径
        min_samples: 最小簇大小
        metric: 距离度量
        
    Returns:
        labels: 聚类标签
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n聚类结果:")
    print(f"  簇数量: {n_clusters}")
    print(f"  噪声点数: {n_noise}")
    
    if n_clusters > 0:
        print(f"  各簇样本数:")
        for label in sorted(set(labels)):
            if label != -1:
                count = list(labels).count(label)
                print(f"    簇 {label}: {count} 个样本")
    
    return labels


def build_sku_database(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    original_features: np.ndarray = None
) -> Dict[str, Any]:
    """
    构建SKU库
    
    Args:
        features: 用于聚类的特征矩阵（已降维）
        labels: 聚类标签
        metadata: 元信息
        original_features: 原始特征矩阵（用于计算簇中心，如果为None则使用features）
        
    Returns:
        SKU数据库
    """
    if original_features is None:
        original_features = features
    
    skus = []
    unique_labels = sorted(set(labels))
    
    sku_id_counter = 1
    
    for label in unique_labels:
        if label == -1:
            continue
        
        mask = labels == label
        cluster_features = features[mask]
        cluster_metadata = [m for m, msk in zip(metadata, mask) if msk]
        
        center = cluster_features.mean(axis=0)
        center_norm = center / (np.linalg.norm(center) + 1e-8)
        
        sku = {
            "sku_id": f"SKU_{sku_id_counter:03d}",
            "cluster_label": int(label),
            "feature_center": center_norm.tolist(),
            "member_count": int(mask.sum()),
            "members": [
                {
                    "image_path": m["image_path"],
                    "bbox": m["bbox"],
                    "confidence": m["confidence"]
                }
                for m in cluster_metadata
            ]
        }
        
        skus.append(sku)
        sku_id_counter += 1
    
    n_noise = list(labels).count(-1)
    
    database = {
        "skus": skus,
        "metadata": {
            "total_boxes": len(labels),
            "total_skus": len(skus),
            "noise_boxes": n_noise,
            "feature_dim": features.shape[1]
        }
    }
    
    print(f"\nSKU库构建完成:")
    print(f"  总SKU数: {len(skus)}")
    print(f"  总箱体数: {len(labels)}")
    print(f"  噪声箱体数: {n_noise}")
    
    return database


def visualize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "SKU Clustering Visualization"
) -> None:
    """
    t-SNE可视化聚类结果
    
    Args:
        features: 特征矩阵
        labels: 聚类标签
        output_path: 输出图片路径
        title: 图表标题
    """
    print(f"\n生成t-SNE可视化...")
    
    n_samples = features.shape[0]
    
    if n_samples < 5:
        print("样本数太少，跳过可视化")
        return
    
    perplexity = min(30, max(5, n_samples // 4))
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000
    )
    
    features_2d = tsne.fit_transform(features)
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    
    fig_width = 16 if n_clusters > 15 else 14
    fig_height = 10
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if n_clusters > 20:
        extra_colors = plt.cm.tab20b(np.linspace(0, 1, n_clusters - 20))
        colors = np.vstack([colors, extra_colors])
    
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c='gray',
                alpha=0.5,
                s=30,
                label=f'Noise ({mask.sum()})',
                marker='x'
            )
        else:
            color_idx = label % len(colors)
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[color_idx]],
                alpha=0.7,
                s=60,
                label=f'SKU_{label+1:03d} ({mask.sum()})',
                edgecolors='white',
                linewidths=0.5
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ncol = 2 if n_clusters > 10 else 1
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=9,
        framealpha=0.9,
        ncol=ncol,
        title='SKU Clusters',
        title_fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"可视化已保存: {output_path}")


def save_sku_database(
    database: Dict[str, Any],
    output_dir: str,
    pca_model: PCA = None
) -> None:
    """
    保存SKU库到文件
    
    Args:
        database: SKU数据库
        output_dir: 输出目录
        pca_model: PCA模型（用于后续匹配）
    """
    import joblib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "sku_database.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"SKU库已保存: {json_path}")
    
    features_path = output_dir / "sku_features.npy"
    feature_centers = np.array([sku["feature_center"] for sku in database["skus"]])
    if len(feature_centers) > 0:
        np.save(features_path, feature_centers)
        print(f"SKU特征已保存: {features_path}")
    
    if pca_model is not None:
        pca_path = output_dir / "pca_model.joblib"
        joblib.dump(pca_model, pca_path)
        print(f"PCA模型已保存: {pca_path}")


def copy_sku_images(
    database: Dict[str, Any],
    output_dir: str
) -> None:
    """
    将SKU成员图片复制到对应文件夹
    
    Args:
        database: SKU数据库
        output_dir: 输出目录
    """
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n复制SKU参考图片...")
    
    for sku in database["skus"]:
        sku_id = sku["sku_id"]
        sku_dir = output_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, member in enumerate(sku["members"]):
            image_path = member.get("image_path", "")
            if not image_path or not Path(image_path).exists():
                continue
            
            src_path = Path(image_path)
            dst_name = f"{src_path.stem}{src_path.suffix}"
            dst_path = sku_dir / dst_name
            
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
        
        print(f"  {sku_id}: 已复制 {len(list(sku_dir.glob('*')))} 张图片")
    
    print(f"SKU参考图片已保存到: {output_dir}")


def load_features(input_path: str) -> List[Dict[str, Any]]:
    """加载特征文件"""
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def print_sku_summary(database: Dict[str, Any]) -> None:
    """打印SKU库摘要"""
    print("\n" + "=" * 70)
    print("SKU库摘要")
    print("=" * 70)
    
    metadata = database["metadata"]
    print(f"总箱体数: {metadata['total_boxes']}")
    print(f"SKU数量: {metadata['total_skus']}")
    print(f"噪声箱体: {metadata['noise_boxes']}")
    print(f"特征维度: {metadata['feature_dim']}")
    
    if database["skus"]:
        print("\n各SKU详情:")
        print("-" * 70)
        for sku in database["skus"]:
            print(f"  {sku['sku_id']}: {sku['member_count']} 个箱体")
    
    print("=" * 70)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='特征聚类模块 - 从箱体特征自动生成SKU库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python sku_clustering.py --input features.pkl --output-dir ./sku_output
    
    # 调整聚类参数
    python sku_clustering.py --input features.pkl --eps 0.4 --min-samples 3
    
    # 调整PCA维度
    python sku_clustering.py --input features.pkl --pca-dim 64
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default="features.pkl",
        help='特征文件路径 (默认: features.pkl)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default="./sku_output",
        help='输出目录 (默认: ./sku_output)'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='DBSCAN邻域半径 (默认: 0.3)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help='DBSCAN最小簇大小 (默认: 2)'
    )
    
    parser.add_argument(
        '--pca-dim',
        type=int,
        default=128,
        help='PCA降维目标维度 (默认: 128)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='不生成可视化图'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 70)
    print("SKU聚类模块")
    print("=" * 70)
    
    print(f"\n[1] 加载特征文件: {args.input}")
    results = load_features(args.input)
    
    print(f"\n[2] 提取特征矩阵")
    features, metadata = extract_features_from_results(results)
    
    print(f"\n[3] L2归一化")
    features_norm = normalize_features(features)
    
    print(f"\n[4] PCA降维")
    features_reduced, pca = reduce_dimensions(features_norm, n_components=args.pca_dim)
    
    print(f"\n[5] DBSCAN聚类 (eps={args.eps}, min_samples={args.min_samples})")
    labels = cluster_features(features_reduced, eps=args.eps, min_samples=args.min_samples)
    
    print(f"\n[6] 构建SKU库")
    database = build_sku_database(features_reduced, labels, metadata)
    
    print_sku_summary(database)
    
    if not args.no_visualize and len(set(labels)) > 1:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_clusters(
            features_reduced, 
            labels, 
            str(output_dir / "cluster_visualization.png")
        )
    
    print(f"\n[7] 保存SKU库")
    save_sku_database(database, args.output_dir, pca_model=pca)
    
    print(f"\n[8] 复制SKU参考图片")
    copy_sku_images(database, str(Path(args.output_dir) / "sku_references"))
    
    print("\n完成!")
    
    return database


if __name__ == '__main__':
    main()
