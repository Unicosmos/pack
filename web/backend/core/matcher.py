"""
SKU匹配器模块 - OML版
实现Ratio Test两步验证的SKU特征匹配
"""

import json
import csv
import os
import sys
import types
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

# 禁用 PyTorch 2.x 的 weights_only 安全检查和新特性
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOADING"] = "False"
os.environ["PYTORCH_JIT"] = "0"

# 创建模拟的 torch.sparse.semi_structured 模块
semi_structured_module = types.ModuleType('torch.sparse.semi_structured')
semi_structured_module.SparseSemiStructuredTensor = None
semi_structured_module.SparseSemiStructuredTensorBCSR = None
semi_structured_module.SparseSemiStructuredTensorBCOO = None
semi_structured_module.SparseSemiStructuredTensorCUSPARSELT = None
semi_structured_module.SparseSemiStructuredTensorCUTLASS = None
semi_structured_module.semi_structured_to_dense = lambda x: x
semi_structured_module.dense_to_semi_structured = lambda x: x
semi_structured_module.to_sparse_semi_structured = lambda x: x
sys.modules['torch.sparse.semi_structured'] = semi_structured_module

try:
    import torch
except ImportError as e:
    print(f"警告: torch导入失败: {e}")
    raise

# Monkey patch torch.load to use weights_only=False by default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SKU"))

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError as e:
    HAS_YOLO = False
    print(f"警告: ultralytics模块导入失败: {e}")

try:
    from feature_extractor import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError as e:
    HAS_FEATURE_EXTRACTOR = False
    print(f"警告: feature_extractor模块导入失败: {e}")

HAS_SKU_MODULES = HAS_YOLO and HAS_FEATURE_EXTRACTOR


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
            model_path: YOLO模型路径
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
            print("警告: FeatureExtractor未加载，匹配功能不可用")
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
            self.sku_features = np.load(str(features_path))
            print(f"  已加载特征矩阵: {self.sku_features.shape}")
        except Exception as e:
            print(f"加载特征文件失败: {e}")
            return

        try:
            self.sku_info = []
            self.sku_labels = []
            with open(index_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.sku_info.append(row)
                    self.sku_labels.append(row['label'])
            print(f"  已加载索引: {len(self.sku_labels)} 条")
        except Exception as e:
            print(f"加载索引文件失败: {e}")
            return

        try:
            self.extractor = FeatureExtractor(model_path=self.sku_model_path, device='cpu')
            print("  FeatureExtractor已初始化")
        except Exception as e:
            print(f"初始化FeatureExtractor失败: {e}")
            return

        self._ready = True
        print(f"  SKUMatcher已就绪")

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
        if not HAS_YOLO:
            print("警告: YOLO模块未加载，检测功能不可用")
            return

        if not Path(self.model_path).exists():
            print(f"警告: 模型文件不存在: {self.model_path}")
            return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                self.detector = YOLO(self.model_path)
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
        return_cropped: bool = True,
        return_plot: bool = False
    ) -> Dict[str, Any]:
        """
        对单张图片执行目标检测

        Args:
            image: PIL Image对象
            return_cropped: 是否返回裁剪图
            return_plot: 是否返回YOLO自带的可视化图片

        Returns:
            检测结果字典
        """
        if not self.is_ready():
            return {"detections": [], "image": image}

        try:
            results = self.detector.predict(
                source=image,
                conf=self.conf_threshold,
                verbose=False
            )

            result = {
                "detections": [],
                "image": image
            }

            # 返回YOLO自带的可视化图片
            if return_plot and len(results) > 0:
                plot_array = results[0].plot()  # YOLO自带可视化，返回numpy数组 (BGR格式)
                plot_image = Image.fromarray(plot_array[..., ::-1])  # BGR转RGB
                result["plot_image"] = plot_image

            if len(results) > 0:
                pred = results[0]
                if pred.boxes is not None and len(pred.boxes) > 0:
                    for i in range(len(pred.boxes)):
                        box = pred.boxes.xyxy[i].cpu().numpy()
                        conf = float(pred.boxes.conf[i].cpu().numpy())
                        cls_id = int(pred.boxes.cls[i].cpu().numpy())
                        cls_name = self.detector.names.get(cls_id, f"class_{cls_id}")

                        x1, y1, x2, y2 = map(int, box)

                        detection = {
                            "bbox": [x1, y1, x2, y2],
                            "class": cls_name,
                            "class_id": cls_id,
                            "confidence": round(conf, 4)
                        }

                        if return_cropped:
                            x1_clamped = max(0, x1)
                            y1_clamped = max(0, y1)
                            x2_clamped = min(image.width, x2)
                            y2_clamped = min(image.height, y2)

                            if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                                cropped = image.crop((x1_clamped, y1_clamped, x2_clamped, y2_clamped))
                                detection["cropped_image"] = cropped
                                detection["cropped_width"] = cropped.width
                                detection["cropped_height"] = cropped.height
                            else:
                                detection["cropped_image"] = None
                                detection["cropped_width"] = 0
                                detection["cropped_height"] = 0

                        result["detections"].append(detection)

            return result
        except Exception as e:
            print(f"检测失败: {e}")
            return {"detections": [], "image": image}
