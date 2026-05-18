from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from PIL import Image

from core.utils.logger import logger

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


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """
    应用EXIF方向信息旋转图片

    Args:
        image: PIL Image对象

    Returns:
        应用了正确方向的图片
    """
    try:
        exif = image.getexif()
        if exif is not None:
            orientation = exif.get(0x0112)

            if orientation == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                image = image.rotate(180)
            elif orientation == 4:
                image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 7:
                image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        logger.debug(f"处理EXIF方向失败: {e}")

    return image


class BoxDetector:
    """YOLO目标检测器封装"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        device: str = None,
        half: bool = False,
        max_det: int = 100,
        classes: List[int] = None
    ):
        """
        初始化检测器

        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值（用于NMS非极大值抑制）
            imgsz: 推理图像尺寸（默认640）
            device: 运行设备，如 'cpu', '0', '0,1,2,3' 等（默认自动选择）
            half: 是否使用半精度推理（GPU加速，需GPU环境）
            max_det: 最大检测数量（默认300）
            classes: 只检测指定类别ID列表，如 [0, 2] 表示只检测第1和第3类（默认None，检测所有类）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.max_det = max_det
        self.classes = classes
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
            logger.info(f"  - 置信度阈值: {self.conf_threshold}")
            logger.info(f"  - IOU阈值: {self.iou_threshold}")
            logger.info(f"  - 图像尺寸: {self.imgsz}")
            logger.info(f"  - 设备: {self.device if self.device else 'auto'}")
            logger.info(f"  - 半精度: {self.half}")
            logger.info(f"  - 最大检测数: {self.max_det}")
            logger.info(f"  - 类别过滤: {self.classes if self.classes else '所有类别'}")
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
            image = apply_exif_orientation(image)

            results = self.detector.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                max_det=self.max_det,
                classes=self.classes,
                verbose=False
            )

            result = {
                "detections": [],
                "image": image
            }

            if return_plot and len(results) > 0:
                plot_array = results[0].plot()
                plot_image = Image.fromarray(plot_array[..., ::-1])
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
