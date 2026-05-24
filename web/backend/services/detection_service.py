"""
检测服务 - 封装YOLO目标检测功能
"""

from typing import Optional, List, Dict, Any
from PIL import Image
from pathlib import Path

from core.detector import BoxDetector
from core.utils.image_utils import (
    filter_small_boxes,
    crop_box,
    resize_with_padding,
    image_to_base64,
    process_uploaded_image,
    generate_crops_base64,
)
from core.utils.logger import logger
from config import config
from schemas.schemas import BoxInfo


class DetectionService:
    """检测服务类"""
    
    def __init__(self):
        self.detector: Optional[BoxDetector] = None
        self._ready = False
    
    def initialize(self):
        """初始化检测器"""
        cfg = config
        
        if cfg.paths.MODEL_PATH.exists():
            logger.info(f"加载检测模型: {cfg.paths.MODEL_PATH}")
            try:
                self.detector = BoxDetector(
                    str(cfg.paths.MODEL_PATH), 
                    conf_threshold=cfg.model.CONF_THRESHOLD
                )
                if self.detector.is_ready():
                    self._ready = True
                    logger.info("  BoxDetector加载成功")
                else:
                    logger.error("  BoxDetector加载失败: 检测器未就绪")
            except Exception as e:
                logger.error(f"  BoxDetector加载失败: {e}")
        else:
            logger.error(f"  错误: 模型文件不存在: {cfg.paths.MODEL_PATH}")
    
    def is_ready(self) -> bool:
        """检查检测器是否就绪"""
        return self._ready and self.detector is not None
    
    def detect(
        self, 
        image_content: bytes, 
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        执行目标检测
        
        Args:
            image_content: 图片内容（字节）
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果字典
        """
        if not self.is_ready():
            raise RuntimeError("检测模型未加载")
        
        image = process_uploaded_image(image_content)
        result = self.detector.detect_single_image(image, return_cropped=True, return_plot=True)
        
        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)
        
        if not boxes:
            return {
                "success": True,
                "count": 0,
                "boxes": [],
                "crops": [],
                "image_with_boxes": None
            }
        
        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )
        
        if plot_image:
            result_image = plot_image
        else:
            from core.visualizer import draw_boxes_only
            result_image = draw_boxes_only(image, boxes)
        img_base64 = image_to_base64(result_image)
        
        crops_base64 = generate_crops_base64(image, boxes, target_size=config.model.INPUT_SIZE)
        
        box_infos = [
            BoxInfo(
                bbox=b.get("bbox", []),
                confidence=b.get("confidence", 0.0),
                class_id=b.get("class_id", 0),
                class_name=b.get("class_name", "box")
            )
            for b in boxes
        ]
        
        return {
            "success": True,
            "count": len(boxes),
            "boxes": box_infos,
            "crops": crops_base64,
            "image_with_boxes": img_base64,
            "image": image
        }
