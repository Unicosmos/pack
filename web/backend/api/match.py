"""
匹配相关API
支持SKU匹配功能
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from config import config
from core.utils.image_utils import process_uploaded_image, crop_box, resize_with_padding
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from models.operation_log import log_operation
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


@router.post("/{task_id}/match", response_model=TaskResponse)
async def match_task(
    task_id: int,
    match_threshold: float = Query(0.85, ge=0, le=1),
    db: Session = Depends(get_db)
):
    """对审核后的检测结果进行SKU匹配"""
    from main import detect_match_service

    if not detect_match_service.is_match_ready():
        raise HTTPException(status_code=503, detail="SKU匹配器未加载")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status not in ["detected", "completed"]:
        raise HTTPException(status_code=400, detail="任务状态不允许匹配")

    try:
        with open(task.image_path, 'rb') as f:
            image = process_uploaded_image(f.read())

        detection_boxes = db.query(DetectionBox).filter(
            DetectionBox.task_id == task_id,
            DetectionBox.status == "approved"
        ).order_by(DetectionBox.box_index).all()

        matches = {}
        matched_count = 0
        unmatched_count = 0

        for db_box in detection_boxes:
            bbox = [db_box.bbox_x1, db_box.bbox_y1, db_box.bbox_x2, db_box.bbox_y2]

            cropped = crop_box(image, bbox)
            if not cropped:
                match_result = MatchResult(
                    box_id=db_box.id,
                    task_id=db_box.task_id,
                    sku_id=None,
                    similarity=None,
                    status="unmatched",
                    top1_sku_id=None
                )
                db.add(match_result)
                matches[f"box_{db_box.box_index}"] = {
                    "status": "unmatched",
                    "sku_id": None,
                    "similarity": None,
                    "top5_labels": []
                }
                unmatched_count += 1
                continue

            custom_sku = db_box.custom_sku

            if custom_sku:
                match_result = MatchResult(
                    box_id=db_box.id,
                    task_id=db_box.task_id,
                    sku_id=custom_sku,
                    sku_name=None,
                    similarity=1.0,
                    status="matched",
                    top1_sku_id=custom_sku,
                    top5_candidates=None
                )
                db.add(match_result)
                
                matches[f"box_{db_box.box_index}"] = {
                    "sku_id": custom_sku,
                    "similarity": 1.0,
                    "status": "matched",
                    "top5_labels": []
                }
                matched_count += 1
            else:
                resized = resize_with_padding(cropped, target_size=detect_match_service.match_service.matcher.input_size)
                features = detect_match_service.match_service.matcher.extract_feature(resized)
                result = detect_match_service.match_service.matcher.match_sku(features, threshold=match_threshold)

                top5_data = []
                if result.top5_labels:
                    for label in result.top5_labels:
                        top5_data.append({
                            "sku_id": label.get("sku_id", ""),
                            "name": label.get("label", ""),
                            "similarity": label.get("similarity", 0),
                            "image_path": label.get("image_path", "")
                        })

                match_result = MatchResult(
                    box_id=db_box.id,
                    task_id=db_box.task_id,
                    sku_id=result.sku_id,
                    sku_name=result.sku_name,
                    similarity=result.similarity,
                    status=result.status,
                    top1_sku_id=result.sku_id,
                    top5_candidates=json.dumps(top5_data) if top5_data else None
                )
                db.add(match_result)

                matches[f"box_{db_box.box_index}"] = {
                    "sku_id": result.sku_id,
                    "similarity": result.similarity,
                    "status": result.status,
                    "top5_labels": top5_data
                }

                if result.status == "matched":
                    matched_count += 1
                else:
                    unmatched_count += 1

        task.matched_count = matched_count
        task.unmatched_count = unmatched_count
        task.status = "completed"
        task.completed_at = datetime.utcnow()

        log_operation(
            db=db,
            entity_type="task",
            entity_id=task_id,
            action="match",
            old_value={"status": "detected"},
            new_value={"status": "completed", "matched": matched_count, "unmatched": unmatched_count}
        )

        db.commit()
        db.refresh(task)

        return task_to_response(task)

    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"匹配失败: {str(e)}")