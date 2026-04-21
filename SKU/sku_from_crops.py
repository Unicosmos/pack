"""
从 crops 目录生成 SKU 库

使用方法:
    # 生成初始 SKU 库
    python sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --update
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
    
    def extract_from_directory(self, directory: str, verbose: bool = True) -> Dict[str, List[np.ndarray]]:
        """
        从目录中的图像提取特征
        
        Args:
            directory: 图像目录
            verbose: 是否显示进度
            
        Returns:
            {sku_name: [feature1, feature2, ...]} 字典
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"错误: 目录不存在 - {directory}")
            return {}
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        sku_features = {}
        
        sku_dirs = [d for d in directory.iterdir() if d.is_dir()]
        total_skus = len(sku_dirs)
        
        if verbose:
            print(f"找到 {total_skus} 个 SKU 目录")
        
        for sku_idx, sku_dir in enumerate(sku_dirs):
            sku_name = sku_dir.name
            
            if verbose:
                print(f"处理 SKU [{sku_idx+1}/{total_skus}]: {sku_name}")
            
            features = []
            image_paths = []
            
            for ext in extensions:
                image_paths.extend(sku_dir.glob(f'*{ext}'))
                image_paths.extend(sku_dir.glob(f'*{ext.upper()}'))
            
            image_paths = sorted(set(image_paths))
            
            if verbose:
                print(f"  找到 {len(image_paths)} 张图片")
            
            for img_path in image_paths:
                try:
                    image = Image.open(img_path)
                    feature = self.extract_single_image(image)
                    features.append(feature)
                except Exception as e:
                    print(f"  警告: 处理失败 - {img_path.name}: {e}")
            
            if features:
                sku_features[sku_name] = features
                if verbose:
                    print(f"  成功提取 {len(features)} 个特征")
            else:
                print(f"  警告: 未提取到特征 - {sku_name}")
        
        return sku_features


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


def build_sku_database(
    sku_features: Dict[str, List[np.ndarray]],
    pca_model: Optional[PCA] = None,
    pca_dim: int = 128
) -> Tuple[Dict[str, Any], np.ndarray, PCA]:
    """
    构建 SKU 库
    
    Args:
        sku_features: {sku_name: [feature1, feature2, ...]} 字典
        pca_model: 已有的 PCA 模型（用于更新）
        pca_dim: PCA 降维目标维度
        
    Returns:
        database: SKU 数据库
        sku_feature_matrix: SKU 特征矩阵
        pca: PCA 模型
    """
    skus = []
    all_features = []
    sku_ids = []
    
    sku_id_counter = 1
    
    for sku_name, features in sku_features.items():
        features_array = np.array(features)
        features_norm = normalize_features(features_array)
        
        # 计算该 SKU 的特征中心
        center = features_norm.mean(axis=0)
        center_norm = center / (np.linalg.norm(center) + 1e-8)
        
        sku = {
            "sku_id": f"SKU_{sku_id_counter:03d}",
            "sku_name": sku_name,
            "feature_center": center_norm.tolist(),
            "member_count": len(features),
            "members": [f"{sku_name}/{i+1}" for i in range(len(features))]
        }
        
        skus.append(sku)
        all_features.append(center_norm)
        sku_ids.append(sku["sku_id"])
        sku_id_counter += 1
    
    # 构建特征矩阵
    sku_feature_matrix = np.array(all_features)
    
    # PCA 降维
    if pca_model is None:
        print(f"\n执行 PCA 降维: {sku_feature_matrix.shape[1]} -> {pca_dim}")
        pca = PCA(n_components=min(pca_dim, sku_feature_matrix.shape[1], sku_feature_matrix.shape[0]))
        sku_feature_matrix = pca.fit_transform(sku_feature_matrix)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"保留方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    else:
        print("\n使用已有 PCA 模型进行降维")
        sku_feature_matrix = pca_model.transform(sku_feature_matrix)
        pca = pca_model
    
    # 重新归一化降维后的特征
    sku_feature_matrix = normalize_features(sku_feature_matrix)
    
    # 更新数据库中的特征中心
    for i, sku in enumerate(skus):
        sku["feature_center"] = sku_feature_matrix[i].tolist()
    
    database = {
        "skus": skus,
        "metadata": {
            "total_boxes": sum(len(features) for features in sku_features.values()),
            "total_skus": len(skus),
            "feature_dim": sku_feature_matrix.shape[1]
        }
    }
    
    print(f"\nSKU库构建完成:")
    print(f"  总SKU数: {len(skus)}")
    print(f"  总箱体数: {database['metadata']['total_boxes']}")
    
    return database, sku_feature_matrix, pca


def save_sku_database(
    database: Dict[str, Any],
    sku_feature_matrix: np.ndarray,
    pca_model: PCA,
    output_dir: str
) -> None:
    """
    保存 SKU 库到文件
    
    Args:
        database: SKU 数据库
        sku_feature_matrix: SKU 特征矩阵
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
    if len(sku_feature_matrix) > 0:
        np.save(features_path, sku_feature_matrix)
        print(f"SKU特征已保存: {features_path}")
    
    pca_path = output_dir / "pca_model.joblib"
    joblib.dump(pca_model, pca_path)
    print(f"PCA模型已保存: {pca_path}")


def load_existing_database(output_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[PCA]]:
    """
    加载已有的 SKU 数据库
    
    Args:
        output_dir: 输出目录
        
    Returns:
        database: SKU 数据库
        pca_model: PCA 模型
    """
    output_dir = Path(output_dir)
    
    database = None
    pca_model = None
    
    # 加载 SKU 数据库
    json_path = output_dir / "sku_database.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        print(f"已加载现有 SKU 库: {json_path}")
    
    # 加载 PCA 模型
    pca_path = output_dir / "pca_model.joblib"
    if pca_path.exists():
        pca_model = joblib.load(pca_path)
        print(f"已加载现有 PCA 模型: {pca_path}")
    
    return database, pca_model


def update_sku_database(
    existing_database: Dict[str, Any],
    new_sku_features: Dict[str, List[np.ndarray]],
    pca_model: PCA
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    更新 SKU 库
    
    Args:
        existing_database: 现有的 SKU 数据库
        new_sku_features: 新的 SKU 特征
        pca_model: PCA 模型
        
    Returns:
        updated_database: 更新后的 SKU 数据库
        updated_feature_matrix: 更新后的特征矩阵
    """
    existing_skus = existing_database.get("skus", [])
    existing_sku_names = {sku.get("sku_name"): sku for sku in existing_skus}
    
    updated_skus = existing_skus.copy()
    all_features = []
    
    # 处理新的 SKU
    for sku_name, features in new_sku_features.items():
        if sku_name in existing_sku_names:
            # 更新现有 SKU
            existing_sku = existing_sku_names[sku_name]
            features_array = np.array(features)
            features_norm = normalize_features(features_array)
            
            # 计算新的特征中心
            new_center = features_norm.mean(axis=0)
            new_center_norm = new_center / (np.linalg.norm(new_center) + 1e-8)
            
            # 应用 PCA
            new_center_pca = pca_model.transform(new_center_norm.reshape(1, -1)).squeeze()
            new_center_pca = new_center_pca / (np.linalg.norm(new_center_pca) + 1e-8)
            
            # 更新 SKU 信息
            existing_sku["feature_center"] = new_center_pca.tolist()
            existing_sku["member_count"] = len(features)
            existing_sku["members"] = [f"{sku_name}/{i+1}" for i in range(len(features))]
            
            print(f"已更新 SKU: {sku_name}")
        else:
            # 添加新 SKU
            features_array = np.array(features)
            features_norm = normalize_features(features_array)
            
            # 计算特征中心
            new_center = features_norm.mean(axis=0)
            new_center_norm = new_center / (np.linalg.norm(new_center) + 1e-8)
            
            # 应用 PCA
            new_center_pca = pca_model.transform(new_center_norm.reshape(1, -1)).squeeze()
            new_center_pca = new_center_pca / (np.linalg.norm(new_center_pca) + 1e-8)
            
            # 创建新 SKU 条目
            new_sku = {
                "sku_id": f"SKU_{len(updated_skus) + 1:03d}",
                "sku_name": sku_name,
                "feature_center": new_center_pca.tolist(),
                "member_count": len(features),
                "members": [f"{sku_name}/{i+1}" for i in range(len(features))]
            }
            
            updated_skus.append(new_sku)
            print(f"已添加新 SKU: {sku_name}")
    
    # 构建更新后的特征矩阵
    for sku in updated_skus:
        all_features.append(sku["feature_center"])
    
    updated_feature_matrix = np.array(all_features)
    
    # 更新数据库
    updated_database = {
        "skus": updated_skus,
        "metadata": {
            "total_boxes": sum(len(features) for features in new_sku_features.values()),
            "total_skus": len(updated_skus),
            "feature_dim": updated_feature_matrix.shape[1]
        }
    }
    
    print(f"\nSKU库更新完成:")
    print(f"  总SKU数: {len(updated_skus)}")
    print(f"  总箱体数: {updated_database['metadata']['total_boxes']}")
    
    return updated_database, updated_feature_matrix


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从 crops 目录生成 SKU 库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 生成初始 SKU 库
    python sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --update
    
    # 指定设备和 PCA 维度
    python sku_from_crops.py --crops-dir ./crops --output-dir ./sku_output --device cuda:0 --pca-dim 64
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
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 70)
    print("从 crops 目录生成 SKU 库")
    print("=" * 70)
    
    # 加载现有数据库（如果更新模式）
    existing_database = None
    existing_pca = None
    
    if args.update:
        print("\n[1] 加载现有 SKU 库")
        existing_database, existing_pca = load_existing_database(args.output_dir)
        
        if existing_database is None:
            print("警告: 未找到现有 SKU 库，将创建新库")
            args.update = False
        elif existing_pca is None:
            print("警告: 未找到现有 PCA 模型，将创建新模型")
            args.update = False
    
    # 初始化特征提取器
    print("\n[2] 初始化特征提取器")
    extractor = FeatureExtractor(device=args.device)
    
    # 提取特征
    print("\n[3] 从 crops 目录提取特征")
    sku_features = extractor.extract_from_directory(args.crops_dir)
    
    if not sku_features:
        print("错误: 未提取到任何特征")
        sys.exit(1)
    
    # 构建或更新 SKU 库
    print("\n[4] 构建 SKU 库")
    if args.update and existing_database and existing_pca:
        database, feature_matrix = update_sku_database(existing_database, sku_features, existing_pca)
        pca_model = existing_pca
    else:
        database, feature_matrix, pca_model = build_sku_database(sku_features, pca_dim=args.pca_dim)
    
    # 保存 SKU 库
    print("\n[5] 保存 SKU 库")
    save_sku_database(database, feature_matrix, pca_model, args.output_dir)
    
    print("\n完成!")
    print(f"SKU 库已生成/更新到: {args.output_dir}")


if __name__ == '__main__':
    main()
