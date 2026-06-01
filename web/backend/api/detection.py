"""
检测相关API
支持触发YOLO检测
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from config import config
from core.utils.image_utils import process_uploaded_image, filter_small_boxes, crop_box, resize_with_padding
from core.visualizer import draw_detection_result
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from schemas.schemas import TaskResponse

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


def task_to_response(task: Task) -> TaskResponse:
    """将Task模型转换为TaskResponse"""
    return TaskResponse(
        id=task.id,
        image_name=task.image_name,
        status=task.status,
        box_count=task.box_count,
        matched_count=task.matched_count,
        unmatched_count=task.unmatched_count,
        vis_image=task.vis_image,
        created_at=task.created_at.isoformat() + 'Z' if task.created_at else "",
        completed_at=task.completed_at.isoformat() + 'Z' if task.completed_at else None
    )


@router.post("/{task_id}/detect", response_model=TaskResponse)
async def detect_task(
    task_id: int,
    match_threshold: float = Query(None, description="匹配阈值"),
    db: Session = Depends(get_db)
):
    """对任务图片执行YOLO检测和SKU匹配"""
    from main import detect_match_service

    detector = detect_match_service.detection_service.detector
    matcher = detect_match_service.match_service.matcher

    if detector is None or not detector.is_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        if task.status != "pending":
            existing_boxes = db.query(DetectionBox).filter(DetectionBox.task_id == task_id).all()
            for box in existing_boxes:
                db.query(MatchResult).filter(MatchResult.box_id == box.id).delete()
                db.delete(box)
            db.commit()

            task.box_count = 0
            task.matched_count = 0
            task.unmatched_count = 0
        with open(task.image_path, 'rb') as f:
            image = process_uploaded_image(f.read())

        result = detector.detect_single_image(image, return_cropped=True, return_plot=True)

        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)

        if not boxes:
            task.status = "completed"
            task.box_count = 0
            task.matched_count = 0
            task.unmatched_count = 0
            task.completed_at = datetime.utcnow()
            db.commit()
            return task_to_response(task)

        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )

        match_results = []
        sku_matcher_enabled = matcher is not None and matcher.is_ready()

        if sku_matcher_enabled and boxes:
            try:
                if match_threshold is None:
                    match_threshold = config.match.MATCH_THRESHOLD
                images_to_match = []
                valid_indices = []
                
                for idx, box in enumerate(boxes):
                    cropped = crop_box(image, box.get("bbox", []))
                    if cropped:
                        resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                        images_to_match.append(resized)
                        valid_indices.append(idx)

                match_results = [None] * len(boxes)
                
                if images_to_match:
                    batch_results = matcher.match_sku_batch(images_to_match, threshold=match_threshold)
                    
                    for i, mr in enumerate(batch_results):
                        original_idx = valid_indices[i]
                        match_results[original_idx] = {
                            'sku_id': mr.sku_id,
                            'sku_name': mr.sku_name,
                            'similarity': mr.similarity,
                            'ratio': mr.ratio,
                            'status': mr.status,
                            'top5_labels': mr.top5_labels if mr.top5_labels else []
                        }
            except Exception as e:
                print(f"匹配失败: {e}")
                match_results = [None] * len(boxes)
        else:
            match_results = [None] * len(boxes)

        crops_dir = config.paths.TASKS_DIR / f"task_{task.id}" / "crops"
        crops_dir.mkdir(exist_ok=True)

        detected_boxes = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.get("bbox", [])
            cropped = image.crop((x1, y1, x2, y2))
            crop_path = crops_dir / f"box_{idx}.jpg"
            cropped.save(crop_path)
            mr = match_results[idx] if idx < len(match_results) else None

            detection_box = DetectionBox(
                task_id=task.id,
                box_index=idx,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                confidence=box.get("confidence", 0.0),
                class_id=box.get("class_id", 0),
                class_name=box.get("class_name", "box"),
                path=str(crop_path),
                status="approved",
                is_audited=False
            )
            db.add(detection_box)
            db.flush()

            top5_data = []
            if mr and mr.get('top5_labels'):
                for label in mr['top5_labels']:
                    top5_data.append({
                        "sku_id": label.get("sku_id", ""),
                        "name": label.get("sku_name", ""),
                        "similarity": label.get("similarity", 0),
                        "image_path": label.get("image_path", "")
                    })

            match_result = MatchResult(
                box_id=detection_box.id,
                task_id=task.id,
                sku_id=mr.get('sku_id') if mr else None,
                sku_name=mr.get('sku_name') if mr else None,
                similarity=mr.get('similarity') if mr else None,
                status=mr.get('status', 'unmatched') if mr else 'unmatched',
                top1_sku_id=mr.get('sku_id') if mr else None,
                top5_candidates=json.dumps(top5_data) if top5_data else None
            )
            db.add(match_result)

            detected_boxes.append({
                "box_id": str(idx),
                "bbox": [x1, y1, x2, y2],
                "confidence": box.get("confidence", 0.0),
                "class_id": box.get("class_id", 0),
                "class_name": box.get("class_name", "box"),
                "status": "approved",
                "is_audited": False,
                "crop_path": str(crop_path)
            })

        matched_count = sum(1 for mr in match_results if mr and mr.get('sku_id'))
        unmatched_count = sum(1 for mr in match_results if mr is None or not mr.get('sku_id'))

        task.box_count = len(detected_boxes)
        task.matched_count = matched_count
        task.unmatched_count = unmatched_count
        task.status = "detected"
        task.completed_at = datetime.utcnow()

        task_dir = config.paths.TASKS_DIR / f"task_{task.id}"
        task_dir.mkdir(exist_ok=True)
        
        try:
            plot_image, _ = draw_detection_result(image, boxes, match_results)
            
            plot_path = task_dir / "detection_result.jpg"
            plot_image.save(plot_path, format='JPEG')
            
            task.vis_image = str(plot_path)
        except Exception as e:
            print(f"生成可视化结果失败: {e}")

        db.commit()
        db.refresh(task)

        return task_to_response(task)

    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
