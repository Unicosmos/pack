"""
审核相关API
支持检测结果的人工审核
"""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from models.sku import SKU
from models.operation_log import log_operation
from schemas.schemas import ReviewUpdate, ReviewResponse

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


@router.put("/{task_id}/review", response_model=ReviewResponse)
async def review_task_detections(
    task_id: int,
    review_data: ReviewUpdate,
    db: Session = Depends(get_db)
):
    """审核任务的检测结果"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status not in ["detected", "completed"]:
        raise HTTPException(status_code=400, detail="任务状态不允许审核")

    try:
        detection_boxes = db.query(DetectionBox).filter(
            DetectionBox.task_id == task_id
        ).order_by(DetectionBox.box_index).all()

        approved_count = 0
        rejected_count = 0
        deleted_count = 0

        review_box_map = {box.get("box_id"): box for box in review_data.boxes}

        for db_box in detection_boxes:
            box_id_str = f"box_{db_box.box_index}"
            
            if box_id_str in review_box_map:
                review_box = review_box_map[box_id_str]
                new_status = review_box.get("status", "approved")

                if "custom_sku" in review_box:
                    db_box.custom_sku = review_box["custom_sku"]

                db_box.status = new_status
                db_box.is_audited = True
                db_box.reviewed_at = datetime.utcnow()

                if new_status == "approved":
                    approved_count += 1
                elif new_status == "rejected":
                    rejected_count += 1
            else:
                deleted_count += 1
                db.query(MatchResult).filter(MatchResult.box_id == db_box.id).delete(synchronize_session=False)
                db.delete(db_box)

        task.box_count = approved_count

        detection_boxes = db.query(DetectionBox).filter(
            DetectionBox.task_id == task_id
        ).order_by(DetectionBox.box_index).all()

        matched_count = 0
        unmatched_count = 0
        
        for db_box in detection_boxes:
            if db_box.status == "approved":
                mr = db.query(MatchResult).filter(
                    MatchResult.box_id == db_box.id
                ).first()
                
                has_custom_sku = db_box.custom_sku
                
                if has_custom_sku:
                    sku = db.query(SKU).filter(SKU.sku_id == has_custom_sku).first()
                    sku_name = sku.sku_name if sku else None
                    
                    if mr:
                        mr.sku_id = has_custom_sku
                        mr.sku_name = sku_name
                        mr.status = "matched"
                        mr.is_manual_override = True
                        mr.override_at = datetime.utcnow()
                        db.add(mr)
                    else:
                        mr = MatchResult(
                            box_id=db_box.id,
                            task_id=db_box.task_id,
                            sku_id=has_custom_sku,
                            sku_name=sku_name,
                            similarity=1.0,
                            status="matched",
                            top1_sku_id=has_custom_sku,
                            is_manual_override=True,
                            override_at=datetime.utcnow()
                        )
                        db.add(mr)
                    matched_count += 1
                elif mr and mr.sku_id:
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
            action="review",
            old_value={"status": "detected"},
            new_value={"status": "completed", "approved": approved_count, "rejected": rejected_count, "deleted": deleted_count}
        )

        db.commit()
        db.refresh(task)

        return ReviewResponse(
            success=True,
            task_id=task_id,
            approved_count=approved_count,
            rejected_count=rejected_count,
            message=f"审核完成：通过 {approved_count} 个，拒绝 {rejected_count} 个，删除 {deleted_count} 个"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"审核失败: {str(e)}")