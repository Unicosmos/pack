"""
从 crops 目录更新 SKU 库

自动匹配新箱体到现有 SKU 或创建新 SKU

使用方法:
    # 初始构建 SKU 库
    python update_sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python update_sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --update
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import joblib


class FeatureExtractor:
    """基于 ResNet50 的特征提取器"""
    
    def __init__(self, device: str = None):
        """
        初始化特征提取器
        
        Args:
            device: 推理设备 (如 'cuda:0', 'cpu')
        """
        self.device = self._get_device(device)
        
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = self._get_preprocess()
        
        print(f"已加载 ResNet50 模型")
        print(f"特征维度: {self.feature_dim}")
        print(f"设备: {self.device}")
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """确定推理设备"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    
    def _load_model(self) -> nn.Module:
        """加载预训练模型并移除分类头"""
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        
        self.feature_dim = model.fc.in_features
        
        backbone = nn.Sequential(*list(model.children())[:-1])
        
        return backbone
    
    def _get_preprocess(self) -> transforms.Compose:
        """获取图像预处理管道"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_single_image(self, image: Image.Image) -> np.ndarray:
        """
        从单张图像提取特征
        
        Args:
            image: PIL.Image 对象
            
        Returns:
            特征向量 (numpy array, shape: [feature_dim])
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def extract_from_crops(self, crops_dir: str, verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        从 crops 目录提取特征
        
        Args:
            crops_dir: crops 目录路径
            verbose: 是否显示进度
            
        Returns:
            [(image_path, feature), ...] 列表
        """
        crops_dir = Path(crops_dir)
        if not crops_dir.exists():
            print(f"错误: 目录不存在 - {crops_dir}")
            return []
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        features_list = []
        
        # 递归遍历所有子目录
        image_paths = []
        for root, dirs, files in os.walk(crops_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    image_paths.append(Path(root) / file)
        
        total_images = len(image_paths)
        
        if verbose:
            print(f"找到 {total_images} 张箱体图片")
        
        for img_idx, img_path in enumerate(image_paths):
            if verbose and img_idx % 10 == 0:
                print(f"  处理 [{img_idx+1}/{total_images}]: {img_path.relative_to(crops_dir)}")
            
            try:
                image = Image.open(img_path)
                feature = self.extract_single_image(image)
                features_list.append((str(img_path), feature))
            except Exception as e:
                print(f"  警告: 处理失败 - {img_path.name}: {e}")
        
        if verbose:
            print(f"成功提取 {len(features_list)} 个特征")
        
        return features_list


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
    
    return normalized


def load_sku_database(output_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[PCA]]:
    """
    加载 SKU 数据库
    
    Args:
        output_dir: 输出目录
        
    Returns:
        database: SKU 数据库
        sku_features: SKU 特征矩阵
        pca_model: PCA 模型
    """
    output_dir = Path(output_dir)
    
    database = None
    sku_features = None
    pca_model = None
    
    # 加载 SKU 数据库
    json_path = output_dir / "sku_database.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        print(f"已加载现有 SKU 库: {json_path}")
    
    # 加载 SKU 特征
    features_path = output_dir / "sku_features.npy"
    if features_path.exists():
        sku_features = np.load(features_path)
        print(f"已加载 SKU 特征: {features_path}")
    
    # 加载 PCA 模型
    pca_path = output_dir / "pca_model.joblib"
    if pca_path.exists():
        pca_model = joblib.load(pca_path)
        print(f"已加载 PCA 模型: {pca_path}")
    
    return database, sku_features, pca_model


def match_to_sku(new_feature: np.ndarray, sku_features: np.ndarray, threshold: float = 0.85) -> Tuple[Optional[int], float]:
    """
    将新特征匹配到现有 SKU
    
    Args:
        new_feature: 新特征向量
        sku_features: SKU 特征矩阵
        threshold: 匹配阈值
        
    Returns:
        (sku_index, similarity) 或 (None, 0.0)
    """
    if sku_features is None or len(sku_features) == 0:
        return None, 0.0
    
    similarities = np.dot(sku_features, new_feature)
    max_idx = np.argmax(similarities)
    max_sim = float(similarities[max_idx])
    
    if max_sim >= threshold:
        return max_idx, max_sim
    else:
        return None, max_sim


def build_initial_sku_database(
    features_list: List[Tuple[str, np.ndarray]],
    pca_dim: int = 128
) -> Tuple[Dict[str, Any], np.ndarray, PCA]:
    """
    构建初始 SKU 库
    
    Args:
        features_list: [(image_path, feature), ...] 列表
        pca_dim: PCA 降维目标维度
        
    Returns:
        database: SKU 数据库
        sku_features: SKU 特征矩阵
        pca_model: PCA 模型
    """
    skus = []
    all_features = []
    
    # 提取所有特征
    features = np.array([f for _, f in features_list])
    features_norm = normalize_features(features)
    
    # PCA 降维
    print(f"\n执行 PCA 降维: {features_norm.shape[1]} -> {pca_dim}")
    pca = PCA(n_components=min(pca_dim, features_norm.shape[1], features_norm.shape[0]))
    features_pca = pca.fit_transform(features_norm)
    features_pca = normalize_features(features_pca)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"保留方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # 初始时每个箱体作为一个 SKU
    for i, (image_path, _) in enumerate(features_list):
        sku = {
            "sku_id": f"SKU_{i+1:03d}",
            "sku_name": f"SKU_{i+1:03d}",
            "feature_center": features_pca[i].tolist(),
            "member_count": 1,
            "members": [image_path]
        }
        skus.append(sku)
        all_features.append(features_pca[i])
    
    sku_features = np.array(all_features)
    
    database = {
        "skus": skus,
        "metadata": {
            "total_boxes": len(features_list),
            "total_skus": len(skus),
            "feature_dim": sku_features.shape[1]
        }
    }
    
    print(f"\n初始 SKU 库构建完成:")
    print(f"  总SKU数: {len(skus)}")
    print(f"  总箱体数: {len(features_list)}")
    
    return database, sku_features, pca


def update_sku_database(
    existing_database: Dict[str, Any],
    existing_sku_features: np.ndarray,
    pca_model: PCA,
    features_list: List[Tuple[str, np.ndarray]],
    threshold: float = 0.85
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    更新 SKU 库
    
    Args:
        existing_database: 现有 SKU 数据库
        existing_sku_features: 现有 SKU 特征矩阵
        pca_model: PCA 模型
        features_list: [(image_path, feature), ...] 列表
        threshold: 匹配阈值
        
    Returns:
        updated_database: 更新后的 SKU 数据库
        updated_sku_features: 更新后的 SKU 特征矩阵
    """
    existing_skus = existing_database.get("skus", [])
    updated_skus = existing_skus.copy()
    
    # 处理每个新特征
    new_skus_count = 0
    matched_count = 0
    
    for image_path, feature in features_list:
        # 归一化并降维
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        feature_pca = pca_model.transform(feature_norm.reshape(1, -1)).squeeze()
        feature_pca = feature_pca / (np.linalg.norm(feature_pca) + 1e-8)
        
        # 匹配到现有 SKU
        sku_idx, similarity = match_to_sku(feature_pca, existing_sku_features, threshold)
        
        if sku_idx is not None:
            # 更新现有 SKU
            matched_sku = updated_skus[sku_idx]
            
            # 添加新成员
            matched_sku["members"].append(image_path)
            matched_sku["member_count"] = len(matched_sku["members"])
            
            # 更新特征中心
            all_member_features = []
            for member_path in matched_sku["members"]:
                # 重新提取特征（这里简化处理，实际应该缓存特征）
                try:
                    image = Image.open(member_path)
                    member_feature = extractor.extract_single_image(image)
                    member_feature_norm = member_feature / (np.linalg.norm(member_feature) + 1e-8)
                    member_feature_pca = pca_model.transform(member_feature_norm.reshape(1, -1)).squeeze()
                    member_feature_pca = member_feature_pca / (np.linalg.norm(member_feature_pca) + 1e-8)
                    all_member_features.append(member_feature_pca)
                except Exception as e:
                    print(f"  警告: 提取成员特征失败 - {member_path}: {e}")
            
            if all_member_features:
                new_center = np.mean(all_member_features, axis=0)
                new_center = new_center / (np.linalg.norm(new_center) + 1e-8)
                matched_sku["feature_center"] = new_center.tolist()
            
            print(f"  已匹配到 SKU {matched_sku['sku_id']} (相似度: {similarity:.4f})")
            matched_count += 1
        else:
            # 创建新 SKU
            new_sku_id = f"SKU_{len(updated_skus) + 1:03d}"
            new_sku = {
                "sku_id": new_sku_id,
                "sku_name": new_sku_id,
                "feature_center": feature_pca.tolist(),
                "member_count": 1,
                "members": [image_path]
            }
            updated_skus.append(new_sku)
            print(f"  已创建新 SKU {new_sku_id} (最高相似度: {similarity:.4f})")
            new_skus_count += 1
    
    # 重建 SKU 特征矩阵
    updated_sku_features = np.array([sku["feature_center"] for sku in updated_skus])
    
    # 更新数据库
    updated_database = {
        "skus": updated_skus,
        "metadata": {
            "total_boxes": sum(sku["member_count"] for sku in updated_skus),
            "total_skus": len(updated_skus),
            "feature_dim": updated_sku_features.shape[1]
        }
    }
    
    print(f"\nSKU 库更新完成:")
    print(f"  总SKU数: {len(updated_skus)}")
    print(f"  总箱体数: {updated_database['metadata']['total_boxes']}")
    print(f"  新增SKU数: {new_skus_count}")
    print(f"  匹配到现有SKU数: {matched_count}")
    
    return updated_database, updated_sku_features


def save_sku_database(
    database: Dict[str, Any],
    sku_features: np.ndarray,
    pca_model: PCA,
    output_dir: str
) -> None:
    """
    保存 SKU 库到文件
    
    Args:
        database: SKU 数据库
        sku_features: SKU 特征矩阵
        pca_model: PCA 模型
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "sku_database.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"SKU库已保存: {json_path}")
    
    features_path = output_dir / "sku_features.npy"
    if len(sku_features) > 0:
        np.save(features_path, sku_features)
        print(f"SKU特征已保存: {features_path}")
    
    pca_path = output_dir / "pca_model.joblib"
    joblib.dump(pca_model, pca_path)
    print(f"PCA模型已保存: {pca_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从 crops 目录更新 SKU 库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 初始构建 SKU 库
    python update_sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python update_sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --update
    
    # 调整匹配阈值和 PCA 维度
    python update_sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --threshold 0.8 --pca-dim 64
        """
    )
    
    parser.add_argument(
        '--crops-dir', '-c',
        type=str,
        default="./crops",
        help='crops 目录路径 (默认: ./crops)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default="./sku_output",
        help='输出目录 (默认: ./sku_output)'
    )
    
    parser.add_argument(
        '--update', '-u',
        action='store_true',
        help='更新现有 SKU 库'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='推理设备 (如 cuda:0, cpu)'
    )
    
    parser.add_argument(
        '--pca-dim',
        type=int,
        default=128,
        help='PCA 降维目标维度 (默认: 128)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='SKU 匹配阈值 (默认: 0.85)'
    )
    
    return parser.parse_args()


# 全局变量，用于更新时的特征提取
extractor = None


def main():
    """
    主函数
    1. 从 crops 目录提取所有箱体特征
    2. 与现有 SKU 库匹配
    3. 更新或创建 SKU
    4. 保存更新后的 SKU 库
    """
    global extractor
    
    args = parse_arguments()
    
    print("=" * 70)
    print("从 crops 目录更新 SKU 库")
    print("=" * 70)
    
    # 初始化特征提取器
    print("\n[1] 初始化特征提取器")
    extractor = FeatureExtractor(device=args.device)
    
    # 从 crops 目录提取特征
    print("\n[2] 从 crops 目录提取特征")
    features_list = extractor.extract_from_crops(args.crops_dir)
    
    if not features_list:
        print("错误: 未提取到任何特征")
        sys.exit(1)
    
    # 加载现有 SKU 库
    print("\n[3] 加载现有 SKU 库")
    existing_database, existing_sku_features, existing_pca = load_sku_database(args.output_dir)
    
    # 构建或更新 SKU 库
    print("\n[4] 构建/更新 SKU 库")
    if args.update and existing_database and existing_sku_features is not None and existing_pca:
        # 更新现有 SKU 库
        database, sku_features = update_sku_database(
            existing_database, 
            existing_sku_features, 
            existing_pca, 
            features_list, 
            threshold=args.threshold
        )
        pca_model = existing_pca
    else:
        # 构建初始 SKU 库
        database, sku_features, pca_model = build_initial_sku_database(
            features_list, 
            pca_dim=args.pca_dim
        )
    
    # 保存 SKU 库
    print("\n[5] 保存 SKU 库")
    save_sku_database(database, sku_features, pca_model, args.output_dir)
    
    print("\n完成!")
    print(f"SKU 库已生成/更新到: {args.output_dir}")


if __name__ == '__main__':
    main()
