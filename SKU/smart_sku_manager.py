"""
智能 SKU 管理器

从 crops 目录智能构建和更新 SKU 库

使用方法:
    # 初始构建 SKU 库
    python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output --update
    
    # 人工调整后更新
    python smart_sku_manager.py --output-dir ./sku_output --sync

    # 调整匹配阈值和 PCA 维度
python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output --threshold 0.8 --pca-dim 64
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

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
    
    def assess_image_quality(self, image: Image.Image) -> Tuple[bool, str]:
        """
        评估图片质量
        
        Args:
            image: PIL.Image 对象
            
        Returns:
            (是否适合作为 SKU, 原因)
        """
        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        
        # 检查图片尺寸
        width, height = image.size
        if width < 50 or height < 50:
            return False, "图片尺寸过小"
        
        # 检查图片比例（放宽限制）
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5.0:
            return False, "图片比例异常"
        
        # 检查亮度
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 20:
            return False, "图片过暗"
        if mean_brightness > 230:
            return False, "图片过亮"
        
        # 检查对比度
        std_brightness = np.std(gray)
        if std_brightness < 5:
            return False, "图片对比度低"
        
        # 检查边缘（放宽限制）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        if edge_density < 0.005:
            return False, "图片边缘特征不足"
        
        return True, ""  
    
    def extract_from_image_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图片路径提取特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            特征向量 或 None
        """
        try:
            image = Image.open(image_path)
            
            # 评估图片质量
            is_quality, reason = self.assess_image_quality(image)
            if not is_quality:
                print(f"  警告: 图片质量不适合作为 SKU - {image_path}: {reason}")
                return None
            
            return self.extract_single_image(image)
        except Exception as e:
            print(f"  警告: 提取特征失败 - {image_path}: {e}")
            return None


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


def match_to_sku(new_feature: np.ndarray, sku_features: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    """
    将新特征匹配到现有 SKU
    
    Args:
        new_feature: 新特征向量
        sku_features: SKU 特征矩阵
        top_k: 返回前 k 个匹配结果
        
    Returns:
        [(sku_index, similarity), ...] 列表
    """
    if sku_features is None or len(sku_features) == 0:
        return []
    
    similarities = np.dot(sku_features, new_feature)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def build_initial_sku_database(
    image_paths: List[str],
    extractor: FeatureExtractor,
    pca_dim: int = 128
) -> Tuple[Dict[str, Any], np.ndarray, PCA]:
    """
    构建初始 SKU 库
    
    Args:
        image_paths: 图片路径列表
        extractor: 特征提取器
        pca_dim: PCA 降维目标维度
        
    Returns:
        database: SKU 数据库
        sku_features: SKU 特征矩阵
        pca_model: PCA 模型
    """
    skus = []
    all_features = []
    
    # 提取所有特征
    features = []
    valid_paths = []
    
    print("\n提取初始特征...")
    for i, img_path in enumerate(image_paths):
        if i % 10 == 0:
            print(f"  处理 [{i+1}/{len(image_paths)}]")
        
        feature = extractor.extract_from_image_path(img_path)
        if feature is not None:
            features.append(feature)
            valid_paths.append(img_path)
    
    if not features:
        print("错误: 未提取到任何特征")
        sys.exit(1)
    
    features = np.array(features)
    features_norm = normalize_features(features)
    
    # PCA 降维
    print(f"\n执行 PCA 降维: {features_norm.shape[1]} -> {pca_dim}")
    pca = PCA(n_components=min(pca_dim, features_norm.shape[1], features_norm.shape[0]))
    features_pca = pca.fit_transform(features_norm)
    features_pca = normalize_features(features_pca)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"保留方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # 智能初始聚类
    print("\n智能初始聚类...")
    processed = set()
    sku_id_counter = 1
    
    for i, (img_path, feature_pca) in enumerate(zip(valid_paths, features_pca)):
        if i in processed:
            continue
        
        # 找到相似的图片
        similarities = np.dot(features_pca, feature_pca)
        similar_indices = [j for j, sim in enumerate(similarities) if sim >= 0.85 and j not in processed]
        
        if similar_indices:
            # 创建 SKU
            members = [valid_paths[j] for j in similar_indices]
            center = np.mean(features_pca[similar_indices], axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            
            sku = {
                "sku_id": f"SKU_{sku_id_counter:03d}",
                "sku_name": f"SKU_{sku_id_counter:03d}",
                "feature_center": center.tolist(),
                "member_count": len(members),
                "members": members
            }
            skus.append(sku)
            all_features.append(center)
            sku_id_counter += 1
            
            # 标记为已处理
            processed.update(similar_indices)
        else:
            # 单个图片作为 SKU
            sku = {
                "sku_id": f"SKU_{sku_id_counter:03d}",
                "sku_name": f"SKU_{sku_id_counter:03d}",
                "feature_center": feature_pca.tolist(),
                "member_count": 1,
                "members": [img_path]
            }
            skus.append(sku)
            all_features.append(feature_pca)
            sku_id_counter += 1
            processed.add(i)
    
    sku_features = np.array(all_features)
    
    database = {
        "skus": skus,
        "metadata": {
            "total_boxes": len(valid_paths),
            "total_skus": len(skus),
            "feature_dim": sku_features.shape[1]
        }
    }
    
    print(f"\n初始 SKU 库构建完成:")
    print(f"  总SKU数: {len(skus)}")
    print(f"  总箱体数: {len(valid_paths)}")
    
    return database, sku_features, pca


def update_sku_database(
    existing_database: Dict[str, Any],
    existing_sku_features: np.ndarray,
    pca_model: PCA,
    new_image_paths: List[str],
    extractor: FeatureExtractor,
    threshold: float = 0.85
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    更新 SKU 库
    
    Args:
        existing_database: 现有 SKU 数据库
        existing_sku_features: 现有 SKU 特征矩阵
        pca_model: PCA 模型
        new_image_paths: 新图片路径列表
        extractor: 特征提取器
        threshold: 匹配阈值
        
    Returns:
        updated_database: 更新后的 SKU 数据库
        updated_sku_features: 更新后的 SKU 特征矩阵
    """
    existing_skus = existing_database.get("skus", [])
    updated_skus = existing_skus.copy()
    
    # 处理每个新图片
    new_skus_count = 0
    matched_count = 0
    
    print("\n处理新图片...")
    for i, img_path in enumerate(new_image_paths):
        if i % 10 == 0:
            print(f"  处理 [{i+1}/{len(new_image_paths)}]: {Path(img_path).name}")
        
        # 提取特征
        feature = extractor.extract_from_image_path(img_path)
        if feature is None:
            continue
        
        # 归一化并降维
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
        feature_pca = pca_model.transform(feature_norm.reshape(1, -1)).squeeze()
        feature_pca = feature_pca / (np.linalg.norm(feature_pca) + 1e-8)
        
        # 匹配到现有 SKU
        matches = match_to_sku(feature_pca, existing_sku_features, top_k=3)
        
        if matches and matches[0][1] >= threshold:
            # 匹配到现有 SKU
            sku_idx, similarity = matches[0]
            matched_sku = updated_skus[sku_idx]
            
            # 添加新成员
            if img_path not in matched_sku["members"]:
                matched_sku["members"].append(img_path)
                matched_sku["member_count"] = len(matched_sku["members"])
                
                # 更新特征中心
                all_member_features = []
                for member_path in matched_sku["members"]:
                    member_feature = extractor.extract_from_image_path(member_path)
                    if member_feature is not None:
                        member_feature_norm = member_feature / (np.linalg.norm(member_feature) + 1e-8)
                        member_feature_pca = pca_model.transform(member_feature_norm.reshape(1, -1)).squeeze()
                        member_feature_pca = member_feature_pca / (np.linalg.norm(member_feature_pca) + 1e-8)
                        all_member_features.append(member_feature_pca)
                
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
                "members": [img_path]
            }
            updated_skus.append(new_sku)
            print(f"  已创建新 SKU {new_sku_id} (最高相似度: {matches[0][1]:.4f} 低于阈值 {threshold})")
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


def sync_sku_database(
    output_dir: str,
    extractor: FeatureExtractor
) -> Tuple[Dict[str, Any], np.ndarray, PCA]:
    """
    同步 SKU 库（人工调整后）
    
    Args:
        output_dir: 输出目录
        extractor: 特征提取器
        
    Returns:
        database: SKU 数据库
        sku_features: SKU 特征矩阵
        pca_model: PCA 模型
    """
    output_dir = Path(output_dir)
    sku_images_dir = output_dir
    
    if not sku_images_dir.exists():
        print(f"错误: SKU 图片目录不存在 - {sku_images_dir}")
        sys.exit(1)
    
    # 第一步：收集所有 SKU 的信息和原始特征
    print("\n同步 SKU 库...")
    sku_info_list = []
    all_raw_features = []
    
    for sku_dir in sku_images_dir.iterdir():
        if not sku_dir.is_dir():
            continue
        
        sku_name = sku_dir.name
        
        # 收集图片
        image_paths = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for ext in extensions:
            image_paths.extend(sku_dir.glob(f'*{ext}'))
            image_paths.extend(sku_dir.glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"  警告: SKU {sku_name} 目录为空")
            continue
        
        # 提取特征
        features = []
        for img_path in image_paths:
            feature = extractor.extract_from_image_path(img_path)
            if feature is not None:
                features.append(feature)
        
        if not features:
            print(f"  警告: SKU {sku_name} 无有效特征")
            continue
        
        # 计算特征中心
        features = np.array(features)
        features_norm = normalize_features(features)
        center = np.mean(features_norm, axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        
        sku_info = {
            "name": sku_name,
            "image_paths": image_paths,
            "raw_feature": center
        }
        sku_info_list.append(sku_info)
        all_raw_features.append(center)
        
        print(f"  已收集 SKU {sku_name}: {len(image_paths)} 张图片")
    
    if not sku_info_list:
        print("错误: 未找到任何 SKU")
        sys.exit(1)
    
    # 第二步：统一进行 PCA 降维
    print("\n构建特征空间...")
    all_raw_features = np.array(all_raw_features)
    
    # 加载或创建 PCA 模型
    pca_path = output_dir / "pca_model.joblib"
    if pca_path.exists():
        pca_model = joblib.load(pca_path)
        print(f"  加载现有 PCA 模型")
    else:
        # 创建新的 PCA 模型
        n_samples = len(all_raw_features)
        n_components = min(128, n_samples - 1, 2048)
        if n_components < 1:
            n_components = 1
        pca_model = PCA(n_components=n_components)
        pca_model.fit(all_raw_features)
        print(f"  创建新的 PCA 模型，组件数: {n_components}")
    
    # 对所有 SKU 特征进行 PCA 降维
    skus = []
    all_features = []
    
    for sku_info in sku_info_list:
        # PCA 降维
        center_pca = pca_model.transform(sku_info["raw_feature"].reshape(1, -1)).squeeze()
        center_pca = center_pca / (np.linalg.norm(center_pca) + 1e-8)
        
        # 使用文件夹名作为 sku_id，保持与现有数据一致
        sku_id = sku_info["name"]
        
        sku = {
            "sku_id": sku_id,
            "sku_name": sku_info["name"],
            "feature_center": center_pca.tolist(),
            "member_count": len(sku_info["image_paths"]),
            "members": sku_info["image_paths"]
        }
        skus.append(sku)
        all_features.append(center_pca)
    
    # 转换为 NumPy 数组
    sku_features = np.array(all_features)
    
    database = {
        "skus": skus,
        "metadata": {
            "total_boxes": sum(sku["member_count"] for sku in skus),
            "total_skus": len(skus),
            "feature_dim": sku_features.shape[1]
        }
    }
    
    print(f"\nSKU 库同步完成:")
    print(f"  总SKU数: {len(skus)}")
    print(f"  总箱体数: {database['metadata']['total_boxes']}")
    print(f"  特征维度: {sku_features.shape[1]}")
    
    return database, sku_features, pca_model


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
    
    # 保存 SKU 数据库
    json_path = output_dir / "sku_database.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    print(f"SKU库已保存: {json_path}")
    
    # 保存 SKU 特征
    features_path = output_dir / "sku_features.npy"
    if len(sku_features) > 0:
        np.save(features_path, sku_features)
        print(f"SKU特征已保存: {features_path}")
    
    # 保存 PCA 模型
    pca_path = output_dir / "pca_model.joblib"
    joblib.dump(pca_model, pca_path)
    print(f"PCA模型已保存: {pca_path}")


def generate_sku_images(
    database: Dict[str, Any],
    output_dir: str
) -> None:
    """
    生成 SKU 图片文件夹
    
    Args:
        database: SKU 数据库
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    sku_images_dir = output_dir 
    sku_images_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n生成 SKU 图片文件夹...")
    
    for sku in database.get("skus", []):
        sku_id = sku["sku_id"]
        sku_dir = sku_images_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图片
        for i, member_path in enumerate(sku["members"]):
            src_path = Path(member_path)
            if not src_path.exists():
                continue
            
            dst_name = f"{i+1}_{src_path.name}"
            dst_path = sku_dir / dst_name
            
            if not dst_path.exists():
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"  警告: 复制图片失败 - {src_path}: {e}")
        
        print(f"  已生成 SKU {sku_id} 图片文件夹: {len(list(sku_dir.glob('*')))} 张图片")


def get_image_paths(crops_dir: str) -> List[str]:
    """
    获取 crops 目录下的所有图片路径
    
    Args:
        crops_dir: crops 目录路径
        
    Returns:
        图片路径列表
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.exists():
        print(f"错误: 目录不存在 - {crops_dir}")
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(crops_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                image_paths.append(str(Path(root) / file))
    
    print(f"找到 {len(image_paths)} 张箱体图片")
    return image_paths


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='智能 SKU 管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 初始构建 SKU 库
    python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output
    
    # 增量更新 SKU 库
    python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output --update
    
    # 人工调整后同步
    python smart_sku_manager.py --output-dir ./sku_output --sync
    
    # 调整参数
    python smart_sku_manager.py --crops-dir ./crops --output-dir ./sku_output --threshold 0.8 --pca-dim 64
        """
    )
    
    parser.add_argument(
        '--crops-dir', '-c',
        type=str,
        help='crops 目录路径'
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
        '--sync', '-s',
        action='store_true',
        help='同步人工调整后的 SKU 库'
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
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='保存同步结果到文件'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_arguments()
    
    print("=" * 70)
    print("智能 SKU 管理器")
    print("=" * 70)
    
    # 初始化特征提取器
    print("\n[1] 初始化特征提取器")
    extractor = FeatureExtractor(device=args.device)
    
    if args.sync:
        # 同步人工调整后的 SKU 库
        print("\n[2] 同步人工调整后的 SKU 库")
        database, sku_features, pca_model = sync_sku_database(args.output_dir, extractor)
        
        if args.save:
            # 保存结果到文件
            print("\n[3] 保存同步结果")
            save_sku_database(database, sku_features, pca_model, args.output_dir)
            print("\n[4] 同步完成（已保存文件）")
        else:
            print("\n[3] 同步完成（仅重新计算特征，不保存文件）")
            print(f"\n如需保存，请使用 --save 选项")
        
        print(f"  总SKU数: {len(database['skus'])}")
        print(f"  总箱体数: {database['metadata']['total_boxes']}")
        return
    else:
        # 获取图片路径
        print("\n[2] 获取图片路径")
        if not args.crops_dir:
            print("错误: 请指定 --crops-dir 参数")
            sys.exit(1)
        image_paths = get_image_paths(args.crops_dir)
        
        if not image_paths:
            print("错误: 未找到任何图片")
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
                image_paths, 
                extractor,
                threshold=args.threshold
            )
            pca_model = existing_pca
        else:
            # 构建初始 SKU 库
            database, sku_features, pca_model = build_initial_sku_database(
                image_paths, 
                extractor,
                pca_dim=args.pca_dim
            )
    
    # 生成 SKU 图片文件夹
    print("\n[5] 生成 SKU 图片文件夹")
    generate_sku_images(database, args.output_dir)
    
    # 保存 SKU 库
    print("\n[6] 保存 SKU 库")
    save_sku_database(database, sku_features, pca_model, args.output_dir)
    
    print("\n完成!")
    print(f"SKU 库已生成/更新到: {args.output_dir}")
    print(f"SKU 图片文件夹: {Path(args.output_dir)}")


if __name__ == '__main__':
    main()
