"""
SKU匹配器模块 - OML版
实现Ratio Test两步验证的SKU特征匹配
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SKU"))

try:
    from box_detector import BoxDetector as OriginalBoxDetector
    from feature_extractor import FeatureExtractor
    HAS_SKU_MODULES = True
except ImportError as e:
    HAS_SKU_MODULES = False
    print(f"警告: SKU模块导入失败: {e}")


@dataclass
class MatchResult:
    sku_id: Optional[str]
    similarity: float
    ratio: Optional[float]
    status: str
    top5_labels: List[Dict[str, float]]


class SKUMatcher:
    """SKU匹配器 - 支持Ratio Test两步验证"""

    def __init__(
        self,
        model_path: str,
        sku_dir: str,
        match_threshold: float = 0.85,
        ratio_threshold: float = 1.2,
        feature_dim: int = 384,
        top_k: int = 5
    ):
        """
        初始化SKU匹配器

        Args:
            model_path: YOLO模型路径
            sku_dir: SKU库目录
            match_threshold: 余弦相似度阈值
            ratio_threshold: Ratio Test阈值
            feature_dim: 特征维度
            top_k: 返回的Top-K候选数
        """
        self.model_path = model_path
        self.sku_dir = Path(sku_dir)
        self.match_threshold = match_threshold
        self.ratio_threshold = ratio_threshold
        self.feature_dim = feature_dim
        self.top_k = top_k

        self.sku_features: Optional[np.ndarray] = None
        self.sku_index: List[Dict[str, str]] = []
        self.label_to_sku: Dict[str, str] = {}

        self._ready = False
        self._load_sku_library()

    def _load_sku_library(self) -> None:
        """加载SKU特征库"""
        if not self.sku_dir.exists():
            print(f"警告: SKU目录不存在: {self.sku_dir}")
            return

        features_path = self.sku_dir / "sku_features.npy"
        index_path = self.sku_dir / "sku_library.csv"

        if not features_path.exists():
            print(f"警告: 特征文件不存在: {features_path}")
            return

        if not index_path.exists():
            print(f"警告: 索引文件不存在: {index_path}")
            return

        try:
            self.sku_features = np.load(features_path)
            print(f"  SKU特征矩阵: {self.sku_features.shape}")

            with open(index_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.sku_index = list(reader)

            for item in self.sku_index:
                label = item.get('label', '')
                sku_id = item.get('sku_id', '')
                if label and sku_id:
                    self.label_to_sku[label] = sku_id

            print(f"  SKU索引数量: {len(self.sku_index)}")
            self._ready = True

        except Exception as e:
            print(f"加载SKU库失败: {e}")

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
        将查询特征与SKU特征库进行匹配（Ratio Test两步验证）

        Args:
            query: L2归一化的查询特征向量 [D]
            threshold: 余弦相似度阈值，默认使用初始化时的值
            ratio_threshold: Ratio Test阈值，默认使用初始化时的值

        Returns:
            MatchResult: 匹配结果
        """
        if threshold is None:
            threshold = self.match_threshold
        if ratio_threshold is None:
            ratio_threshold = self.ratio_threshold

        if not self.is_ready():
            return MatchResult(
                sku_id=None,
                similarity=0.0,
                ratio=None,
                status="unmatched",
                top5_labels=[]
            )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        norms = np.linalg.norm(query, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        query = query / norms

        similarities = np.dot(query, self.sku_features.T)[0]

        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        top5_labels = []
        for idx in top_indices:
            if idx < len(self.sku_index):
                label = self.sku_index[idx].get('label', '')
                top5_labels.append({
                    'label': label,
                    'similarity': float(similarities[idx])
                })

        if not top5_labels:
            return MatchResult(
                sku_id=None,
                similarity=0.0,
                ratio=None,
                status="unmatched",
                top5_labels=[]
            )

        top1_label = top5_labels[0]['label']
        top1_sim = top5_labels[0]['similarity']
        top1_sku = self.label_to_sku.get(top1_label, '')

        first_different_sku_sim = None
        for item in top5_labels:
            label = item['label']
            sku = self.label_to_sku.get(label, '')
            if sku != top1_sku:
                first_different_sku_sim = item['similarity']
                break

        if first_different_sku_sim is None:
            ratio = float('inf')
        else:
            ratio = top1_sim / first_different_sku_sim if first_different_sku_sim > 0 else float('inf')

        if top1_sim < threshold:
            status = "unmatched"
            sku_id = None
        elif ratio < ratio_threshold:
            status = "low_conf"
            sku_id = top1_sku
        else:
            status = "matched"
            sku_id = top1_sku

        return MatchResult(
            sku_id=sku_id,
            similarity=float(top1_sim),
            ratio=float(ratio) if ratio != float('inf') else None,
            status=status,
            top5_labels=top5_labels
        )

    def match_batch(
        self,
        queries: np.ndarray,
        threshold: Optional[float] = None,
        ratio_threshold: Optional[float] = None
    ) -> List[MatchResult]:
        """
        批量匹配

        Args:
            queries: 查询特征矩阵 [N, D]
            threshold: 余弦相似度阈值
            ratio_threshold: Ratio Test阈值

        Returns:
            List[MatchResult]: 匹配结果列表
        """
        return [self.match_sku(q, threshold, ratio_threshold) for q in queries]

    def get_sku_info(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """
        获取SKU信息

        Args:
            sku_id: SKU编号

        Returns:
            SKU信息字典，不存在则返回None
        """
        sku_items = [item for item in self.sku_index if item.get('sku_id') == sku_id]
        if not sku_items:
            return None

        sku_name = sku_items[0].get('sku_name', sku_id)
        image_count = len(sku_items)
        labels = [item.get('label', '') for item in sku_items]

        return {
            'sku_id': sku_id,
            'sku_name': sku_name,
            'image_count': image_count,
            'labels': labels
        }

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        从图像提取特征

        Args:
            image: PIL Image对象

        Returns:
            特征向量 [D]
        """
        if not HAS_SKU_MODULES:
            return np.zeros(self.feature_dim)

        try:
            extractor = FeatureExtractor(device='cpu')
            feat = extractor.extract(image)
            if feat.ndim == 2:
                feat = feat.squeeze()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            return feat
        except Exception as e:
            print(f"特征提取失败: {e}")
            return np.zeros(self.feature_dim)


class BoxDetector:
    """YOLO目标检测器封装"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        初始化检测器

        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.detector = None
        self._ready = False
        self._load_model()

    def _load_model(self) -> None:
        """加载模型"""
        if not HAS_SKU_MODULES:
            print("警告: SKU模块未加载，检测功能不可用")
            return

        if not Path(self.model_path).exists():
            print(f"警告: 模型文件不存在: {self.model_path}")
            return

        try:
            self.detector = OriginalBoxDetector(self.model_path, conf_threshold=self.conf_threshold)
            self._ready = True
            print(f"  BoxDetector已加载: {self.model_path}")
        except Exception as e:
            print(f"加载BoxDetector失败: {e}")

    def is_ready(self) -> bool:
        """检查检测器是否就绪"""
        return self._ready

    def detect_single_image(
        self,
        image: Image.Image,
        return_cropped: bool = True
    ) -> Dict[str, Any]:
        """
        对单张图片执行目标检测

        Args:
            image: PIL Image对象
            return_cropped: 是否返回裁剪图

        Returns:
            检测结果字典
        """
        if not self.is_ready():
            return {"detections": [], "image": image}

        try:
            result = self.detector.detect_single_image(image, return_cropped=return_cropped)
            return result
        except Exception as e:
            print(f"检测失败: {e}")
            return {"detections": [], "image": image}
