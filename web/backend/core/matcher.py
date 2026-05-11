"""
SKU匹配器模块 - OML版
实现Ratio Test两步验证的SKU特征匹配
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

# 初始化PyTorch环境
try:
    from .pytorch_utils import init_pytorch_env
    init_pytorch_env()
except ImportError:
    pass

# 初始化日志
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import logger

try:
    import torch
except ImportError as e:
    logger.error(f"torch导入失败: {e}")
    raise

# 添加SKU目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "SKU"))

try:
    from feature_extractor import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError as e:
    HAS_FEATURE_EXTRACTOR = False
    logger.warning(f"feature_extractor模块导入失败: {e}")


@dataclass
class MatchResult:
    sku_id: Optional[str]
    similarity: float
    ratio: Optional[float]
    status: str
    top5_labels: List[Dict[str, Any]]


class SKUMatcher:
    """SKU特征匹配器"""

    def __init__(
        self,
        model_path: str,
        sku_dir: str,
        feature_dim: int = 384,
        match_threshold: float = 0.85,
        ratio_threshold: float = 1.2,
        top_k: int = 5,
        sku_model_path: Optional[str] = None
    ):
        """
        初始化SKU匹配器

        Args:
            model_path: YOLO模型路径（保持兼容性，实际未使用）
            sku_dir: SKU库目录
            feature_dim: 特征维度
            match_threshold: 相似度阈值
            ratio_threshold: Ratio Test阈值
            top_k: 返回前k个候选
            sku_model_path: SKU微调模型路径 (.pth文件)
        """
        self.model_path = model_path
        self.sku_dir = Path(sku_dir)
        self.feature_dim = feature_dim
        self.match_threshold = match_threshold
        self.ratio_threshold = ratio_threshold
        self.top_k = top_k
        self.sku_model_path = sku_model_path

        self.sku_features = None
        self.sku_labels = None
        self.sku_info = None
        self.extractor = None
        self._ready = False

        self._load_sku_library()

    def _load_sku_library(self) -> None:
        """加载SKU特征库"""
        if not HAS_FEATURE_EXTRACTOR:
            logger.warning("FeatureExtractor未加载，匹配功能不可用")
            return

        features_path = self.sku_dir / "sku_features.npy"
        index_path = self.sku_dir / "sku_library.csv"

        if not features_path.exists():
            logger.warning(f"特征文件不存在: {features_path}")
            return

        if not index_path.exists():
            logger.warning(f"索引文件不存在: {index_path}")
            return

        try:
            self.sku_features = np.load(str(features_path))
            logger.info(f"已加载特征矩阵: {self.sku_features.shape}")
        except Exception as e:
            logger.error(f"加载特征文件失败: {e}")
            return

        try:
            self.sku_info = []
            self.sku_labels = []
            with open(index_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.sku_info.append(row)
                    self.sku_labels.append(row['label'])
            logger.info(f"已加载索引: {len(self.sku_labels)} 条")
        except Exception as e:
            logger.error(f"加载索引文件失败: {e}")
            return

        try:
            self.extractor = FeatureExtractor(model_path=self.sku_model_path, device='cpu')
            logger.info("FeatureExtractor已初始化")
        except Exception as e:
            logger.error(f"初始化FeatureExtractor失败: {e}")
            return

        self._ready = True
        logger.info("SKUMatcher已就绪")

    def is_ready(self) -> bool:
        """检查匹配器是否已就绪"""
        return self._ready and self.sku_features is not None

    def match_sku(
        self,
        query: np.ndarray,
        threshold: Optional[float] = None,
        ratio_threshold: Optional[float] = None
    ) -> MatchResult:
        """
        将查询特征与SKU特征库进行匹配

        Args:
            query: 查询特征向量
            threshold: 相似度阈值
            ratio_threshold: Ratio Test阈值

        Returns:
            MatchResult: 匹配结果
        """
        if not self.is_ready():
            return MatchResult(
                sku_id=None,
                similarity=0.0,
                ratio=None,
                status="unmatched",
                top5_labels=[]
            )

        t = threshold if threshold is not None else self.match_threshold
        rt = ratio_threshold if ratio_threshold is not None else self.ratio_threshold

        query_norm = query / np.linalg.norm(query) if np.linalg.norm(query) > 0 else query

        similarities = np.dot(self.sku_features, query_norm)

        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        top_similarities = similarities[top_indices]
        top_labels = [self.sku_labels[i] for i in top_indices]

        top5_labels = []
        for i, idx in enumerate(top_indices):
            label = self.sku_labels[idx]
            info = self.sku_info[idx] if idx < len(self.sku_info) else {}
            top5_labels.append({
                "label": label,
                "similarity": float(top_similarities[i]),
                "image_name": info.get("image_name", ""),
                "sku_id": info.get("sku_id", ""),
                "sku_name": info.get("sku_name", "")
            })

        if len(top_labels) == 0:
            return MatchResult(
                sku_id=None,
                similarity=0.0,
                ratio=None,
                status="unmatched",
                top5_labels=top5_labels
            )

        top1_label = top_labels[0]
        top1_sim = float(top_similarities[0])

        top1_sku = top1_label.split('_')[0] if '_' in top1_label else top1_label

        first_different_sku_sim = None
        for i in range(1, len(top_labels)):
            sku = top_labels[i].split('_')[0] if '_' in top_labels[i] else top_labels[i]
            if sku != top1_sku:
                first_different_sku_sim = float(top_similarities[i])
                break

        if first_different_sku_sim is None:
            ratio = float('inf')
        else:
            ratio = top1_sim / first_different_sku_sim if first_different_sku_sim > 0 else float('inf')

        if top1_sim < t:
            status = "unmatched"
            sku_id = None
        elif ratio < rt:
            status = "low_conf"
            sku_id = top1_sku
        else:
            status = "matched"
            sku_id = top1_sku

        return MatchResult(
            sku_id=sku_id,
            similarity=top1_sim,
            ratio=float(ratio) if ratio != float('inf') else None,
            status=status,
            top5_labels=top5_labels
        )

    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """
        从图像提取特征

        Args:
            image: PIL Image对象

        Returns:
            特征向量 [D]
        """
        if not HAS_FEATURE_EXTRACTOR or not self.extractor:
            return np.zeros(self.feature_dim)

        try:
            feat = self.extractor.extract(image)
            if feat.ndim == 2:
                feat = feat.squeeze()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            return feat
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return np.zeros(self.feature_dim)
