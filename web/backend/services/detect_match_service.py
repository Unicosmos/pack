"""
检测匹配服务 - 封装检测+匹配的完整业务流程
"""

from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from datetime import datetime
import uuid
from pathlib import Path

from core.visualizer import draw_detection_result
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
from schemas.schemas import BoxInfo, MatchInfo
from models.task import Task
from database import SessionLocal
from .detection_service import DetectionService
from .match_service import MatchService


class DetectMatchService:
    """检测匹配复合服务"""
    
    def __init__(self):
        self.detection_service = DetectionService()
        self.match_service = MatchService()
    
    def initialize(self):
        """初始化所有服务"""
        self.detection_service.initialize()
        self.match_service.initialize()
    
    def is_detection_ready(self) -> bool:
        """检测服务是否就绪"""
        return self.detection_service.is_ready()
    
    def is_match_ready(self) -> bool:
        """匹配服务是否就绪"""
        return self.match_service.is_ready()
    
    def detect_and_match(
        self, 
        contents: bytes, 
        file_name: str,
        conf_threshold: float = 0.5,
        match_threshold: float = None
    ) -> Dict[str, Any]:
        """
        执行检测+匹配完整流程
        
        Args:
            contents: 图片内容（字节）
            file_name: 文件名
            conf_threshold: 检测置信度阈值
            match_threshold: 匹配相似度阈值
            
        Returns:
            完整的检测匹配结果
        """
        if not self.is_detection_ready():
            raise RuntimeError("检测模型未加载")
        
        image = process_uploaded_image(contents)
        
        # 执行检测
        result = self.detection_service.detector.detect_single_image(
            image, return_cropped=True, return_plot=True
        )
        
        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)
        
        if not boxes:
            return {
                "success": True,
                "count": 0,
                "matched_count": 0,
                "unmatched_count": 0,
                "boxes": [],
                "matches": [],
                "crops": [],
                "image_with_boxes": None,
                "sku_matcher_enabled": self.is_match_ready(),
                "task_id": None
            }
        
        # 过滤小框
        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )
        
        # 执行匹配
        match_results = []
        sku_matcher_enabled = self.is_match_ready()
        
        if sku_matcher_enabled and boxes:
            try:
                if match_threshold is None:
                    match_threshold = config.match.MATCH_THRESHOLD
                features = []
                for box in boxes:
                    cropped = crop_box(image, box.get("bbox", []))
                    if cropped:
                        resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                        feat = self.match_service.matcher.extract_feature(resized)
                        features.append(feat)
                    else:
                        features.append(None)
                
                from core.matcher import MatchResult
                for feat in features:
                    if feat is None:
                        match_results.append(MatchResult(
                            sku_id=None,
                            sku_name=None,
                            similarity=0.0,
                            ratio=None,
                            status="unmatched",
                            top5_labels=[]
                        ))
                    else:
                        mr = self.match_service.matcher.match_sku(feat, threshold=match_threshold)
                        match_results.append(mr)
            except Exception as e:
                logger.error(f"匹配失败: {e}")
                sku_matcher_enabled = False
        
        if not sku_matcher_enabled:
            match_results = [None] * len(boxes)
        
        # 生成结果图片
        if plot_image:
            result_image = plot_image
        else:
            result_image, _ = draw_detection_result(image, boxes, match_results)
        img_base64 = image_to_base64(result_image)
        
        # 生成裁剪图
        crops_base64 = generate_crops_base64(image, boxes, target_size=config.model.INPUT_SIZE)
        
        # 构建返回数据
        box_infos = [
            BoxInfo(
                bbox=b.get("bbox", []),
                confidence=b.get("confidence", 0.0),
                class_id=b.get("class_id", 0),
                class_name=b.get("class_name", "box")
            )
            for b in boxes
        ]
        
        match_infos = []
        matched_count = 0
        unmatched_count = 0
        
        for mr in match_results:
            if mr is None:
                match_infos.append(None)
                unmatched_count += 1
            else:
                # 直接传递字典列表，不转换为 TopLabel 对象
                match_infos.append(MatchInfo(
                    sku_id=mr.sku_id,
                    similarity=mr.similarity,
                    ratio=mr.ratio,
                    status=mr.status,
                    top5_labels=mr.top5_labels
                ))
                
                if mr.status == "matched":
                    matched_count += 1
                else:
                    unmatched_count += 1
        
        # 保存任务到数据库
        task_id = self._save_task(file_name, contents, box_infos, match_infos, img_base64, len(boxes))
        
        return {
            "success": True,
            "count": len(boxes),
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "boxes": box_infos,
            "crops": crops_base64,
            "image_with_boxes": img_base64,
            "task_id": task_id,
            "matches": match_infos,
            "sku_matcher_enabled": sku_matcher_enabled
        }
    
    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        执行目标检测（不保存到数据库）

        Args:
            image: PIL Image对象
            conf_threshold: 置信度阈值

        Returns:
            检测结果字典，包含boxes、plot_image等
        """
        if not self.is_detection_ready():
            raise RuntimeError("检测模型未加载")

        result = self.detection_service.detector.detect_single_image(
            image, return_cropped=True, return_plot=True
        )

        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)

        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )

        return {
            "boxes": boxes,
            "plot_image": plot_image,
            "count": len(boxes)
        }

    def match(
        self,
        image: Image.Image,
        boxes: List[Dict],
        match_threshold: float = None
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        对检测结果进行SKU匹配（不保存到数据库）

        Args:
            image: PIL Image对象
            boxes: 检测框列表
            match_threshold: 匹配相似度阈值

        Returns:
            (match_results, matched_count, unmatched_count)
        """
        if not boxes:
            return [], 0, 0

        match_results = []
        sku_matcher_enabled = self.is_match_ready()

        if sku_matcher_enabled:
            if match_threshold is None:
                match_threshold = config.match.MATCH_THRESHOLD
            features = []
            for box in boxes:
                cropped = crop_box(image, box.get("bbox", []))
                if cropped:
                    resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                    feat = self.match_service.matcher.extract_feature(resized)
                    features.append(feat)
                else:
                    features.append(None)

            for feat in features:
                if feat is None:
                    match_results.append({
                        'sku_id': None,
                        'sku_name': None,
                        'similarity': 0.0,
                        'status': 'unmatched',
                        'top5_labels': []
                    })
                else:
                    mr = self.match_service.matcher.match_sku(feat, threshold=match_threshold)
                    match_results.append({
                        'sku_id': mr.sku_id,
                        'sku_name': mr.sku_name,
                        'similarity': mr.similarity,
                        'status': mr.status,
                        'top5_labels': mr.top5_labels if mr.top5_labels else []
                    })
        else:
            match_results = [None] * len(boxes)

        matched_count = sum(1 for mr in match_results if mr and mr.get('status') == 'matched')
        unmatched_count = sum(1 for mr in match_results if mr is None or mr.get('status') != 'matched')

        return match_results, matched_count, unmatched_count

    def _save_task(
        self, 
        file_name: str, 
        contents: bytes, 
        box_infos: List, 
        match_infos: List, 
        img_base64: str, 
        box_count: int
    ) -> Optional[int]:
        """保存任务到数据库"""
        db = SessionLocal()
        try:
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{file_name}"
            upload_dir = config.paths.DATA_DIR / "uploads"
            upload_dir.mkdir(exist_ok=True)
            file_path = upload_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            task = Task(
                task_name=file_name,
                image_name=file_name,
                image_path=str(file_path),
                status="detected",
                result={
                    "detections": {
                        "boxes": [b.dict() for b in box_infos]
                    },
                    "matches": [m.dict() if m else None for m in match_infos],
                    "image_with_boxes": img_base64
                },
                box_count=box_count,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            return task.id
        except Exception as e:
            logger.error(f"保存任务失败: {e}")
            return None
        finally:
            db.close()
    
    def get_sku_list(self) -> List[Dict]:
        """获取SKU列表"""
        return self.match_service.get_sku_list()
    
    def get_sku_count(self) -> int:
        """获取SKU数量"""
        return self.match_service.get_sku_count()
