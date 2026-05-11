"""
YOLO目标检测器模块
封装YOLO目标检测功能，提供箱体检测能力
"""

from pathlib import Path
from typing import Dict, Any, Optional

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

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError as e:
    HAS_YOLO = False
    logger.warning(f"ultralytics模块导入失败: {e}")


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
            logger.warning("YOLO模块未加载，检测功能不可用")
            return

        if not Path(self.model_path).exists():
            logger.warning(f"模型文件不存在: {self.model_path}")
            return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                self.detector = YOLO(self.model_path)
            self._ready = True
            logger.info(f"BoxDetector已加载: {self.model_path}")
        except Exception as e:
            logger.error(f"加载BoxDetector失败: {e}")

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
            logger.error(f"检测失败: {e}")
            return {"detections": [], "image": image}
