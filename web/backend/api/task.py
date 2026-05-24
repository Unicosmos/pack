"""
任务管理API
支持批量检测任务的创建、状态追踪、检测审核和SKU匹配
"""

import os
import uuid
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from database import get_db
from config import config
from core.utils.image_utils import process_uploaded_image, filter_small_boxes, image_to_base64, crop_box, resize_with_padding
from core.visualizer import draw_detection_result, draw_detection_result_from_db
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from models.sku import SKU
from models.operation_log import log_operation
from schemas.schemas import (
    TaskResponse,
    TaskUpdate,
    ReviewUpdate,
    ReviewResponse,
    DetectedBox,
)

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


def get_upload_dir() -> Path:
    """获取上传目录（从config获取）"""
    upload_dir = config.paths.DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


class TaskListResponse(BaseModel):
    success: bool
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int


class BatchTaskResponse(BaseModel):
    success: bool
    task_ids: List[int]
    total_count: int
    message: str


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
        created_at=task.created_at.isoformat() if task.created_at else "",
        completed_at=task.completed_at.isoformat() if task.completed_at else None
    )


@router.post("/upload", response_model=TaskResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """上传图片并创建任务"""
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="只支持图片格式：jpg, png, bmp")

    safe_filename = file.filename.replace('/', '_').replace('\\', '_').replace(':', '_')

    task = Task(
        task_name=file.filename,
        image_name=file.filename,
        image_path="",
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    try:
        task_dir = config.paths.TASKS_DIR / f"task_{task.id}"
        original_dir = task_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        image = process_uploaded_image(content)
        
        file_path = original_dir / safe_filename
        image.save(file_path, format='JPEG', quality=95)

        task.image_path = str(file_path)
        db.commit()
        db.refresh(task)
    except Exception as e:
        db.delete(task)
        db.commit()
        raise HTTPException(status_code=500, detail=f"保存文件失败：{str(e)}")

    return task_to_response(task)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取任务列表"""
    query = db.query(Task)

    if status_filter:
        query = query.filter(Task.status == status_filter)

    total = query.count()
    tasks = query.order_by(Task.created_at.desc()) \
        .offset((page - 1) * page_size) \
        .limit(page_size) \
        .all()

    return TaskListResponse(
        success=True,
        tasks=[task_to_response(t) for t in tasks],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """获取任务详情"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return task_to_response(task)


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    update_data: TaskUpdate,
    db: Session = Depends(get_db)
):
    """更新任务"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if update_data.status is not None:
        task.status = update_data.status
    if update_data.vis_image is not None:
        task.vis_image = update_data.vis_image
    if update_data.box_count is not None:
        task.box_count = update_data.box_count
    if update_data.matched_count is not None:
        task.matched_count = update_data.matched_count
    if update_data.unmatched_count is not None:
        task.unmatched_count = update_data.unmatched_count
    if update_data.error_message is not None:
        task.error_message = update_data.error_message

    if update_data.status == "completed":
        task.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(task)
    return task_to_response(task)


@router.post("/{task_id}/detect", response_model=TaskResponse)
async def detect_task(
    task_id: int,
    match_threshold: float = Query(0.85, description="匹配阈值"),
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
            existing_boxes = db.query(DetectionBox).filter(DetectionBox.task_id == task.id).all()
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


@router.get("/{task_id}/image")
async def get_task_image(
    task_id: int,
    db: Session = Depends(get_db)
):
    """获取任务原图"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not os.path.exists(task.image_path):
        raise HTTPException(status_code=404, detail="图片不存在")

    return FileResponse(task.image_path)


@router.get("/{task_id}/detection-image")
async def get_task_detection_image(
    task_id: int,
    db: Session = Depends(get_db)
):
    """获取任务检测结果可视化图片"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not task.vis_image or not os.path.exists(task.vis_image):
        raise HTTPException(status_code=404, detail="检测结果图片不存在")

    return FileResponse(task.vis_image)

@router.get("/{task_id}/detections")
async def get_task_detections(
    task_id: int,
    db: Session = Depends(get_db)
):
    """获取任务的检测结果（包含匹配数据）"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    detection_boxes = db.query(DetectionBox).filter(
        DetectionBox.task_id == task_id
    ).order_by(DetectionBox.box_index).all()

    if not detection_boxes:
        return {
            "success": True,
            "task_id": task_id,
            "status": task.status,
            "boxes": []
        }

    box_ids = [box.id for box in detection_boxes]
    match_results = db.query(MatchResult).filter(
        MatchResult.box_id.in_(box_ids)
    ).all()
    
    match_map = {mr.box_id: mr for mr in match_results}

    boxes = []
    for db_box in detection_boxes:
        box_data = {
            "box_id": f"box_{db_box.box_index}",
            "bbox": [db_box.bbox_x1, db_box.bbox_y1, db_box.bbox_x2, db_box.bbox_y2],
            "confidence": db_box.confidence,
            "class_id": db_box.class_id,
            "class_name": db_box.class_name,
            "status": db_box.status,
            "is_audited": db_box.is_audited,
            "crop_path": db_box.path,
            "custom_sku": db_box.custom_sku
        }

        mr = match_map.get(db_box.id)
        if mr:
            box_data["match_result"] = {
                "sku_id": mr.sku_id,
                "sku_name": mr.sku_name,
                "similarity": mr.similarity,
                "status": mr.status,
                "top1_sku_id": mr.top1_sku_id,
                "top5_labels": json.loads(mr.top5_candidates) if mr.top5_candidates else []
            }

        boxes.append(box_data)

    return {
        "success": True,
        "task_id": task_id,
        "status": task.status,
        "box_count": task.box_count,
        "matched_count": task.matched_count,
        "unmatched_count": task.unmatched_count,
        "boxes": boxes
    }


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

        for idx, db_box in enumerate(detection_boxes):
            box_id_str = f"box_{idx}"
            
            if box_id_str in review_box_map:
                review_box = review_box_map[box_id_str]
                new_status = review_box.get("status", "approved")
                old_status = db_box.status

                # 保存自定义SKU
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


@router.post("/{task_id}/match", response_model=TaskResponse)
async def match_task(
    task_id: int,
    match_threshold: float = Query(0.85, ge=0, le=1),
    db: Session = Depends(get_db)
):
    """对审核后的检测结果进行SKU匹配"""
    from main import detect_match_service
    from core.utils.image_utils import process_uploaded_image, crop_box, resize_with_padding

    matcher = detect_match_service.match_service.matcher
    detector = detect_match_service.detection_service.detector

    if matcher is None or not matcher.is_ready():
        raise HTTPException(status_code=503, detail="SKU匹配器未加载")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status not in ["detected", "completed"]:
        raise HTTPException(status_code=400, detail="任务状态不允许审核")

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

            # 检查是否有自定义SKU
            custom_sku = db_box.custom_sku

            if custom_sku:
                # 如果有自定义SKU，直接使用
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
                # 否则进行正常匹配
                resized = resize_with_padding(cropped, target_size=matcher.input_size)
                features = matcher.extract_feature(resized)
                result = matcher.match_sku(features, threshold=match_threshold)

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


@router.delete("/{task_id}")
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """删除任务"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    db.query(MatchResult).filter(MatchResult.task_id == task_id).delete(synchronize_session=False)
    db.query(DetectionBox).filter(DetectionBox.task_id == task_id).delete()

    if os.path.exists(task.image_path):
        os.remove(task.image_path)

    task_dir = config.paths.TASKS_DIR / f"task_{task_id}"
    if task_dir.exists():
        import shutil
        shutil.rmtree(task_dir)

    db.delete(task)
    db.commit()

    return {"success": True, "message": "任务已删除"}


@router.get("/stats/summary")
async def get_task_stats(
    db: Session = Depends(get_db)
):
    """获取任务统计"""
    total = db.query(Task).count()
    completed = db.query(Task).filter(Task.status == "completed").count()
    pending = db.query(Task).filter(Task.status == "pending").count()
    detected = db.query(Task).filter(Task.status == "detected").count()
    failed = db.query(Task).filter(Task.status == "failed").count()

    total_detections = db.query(Task).with_entities(func.sum(Task.box_count)).scalar() or 0

    return {
        "success": True,
        "total": total,
        "completed": completed,
        "pending": pending,
        "detected": detected,
        "failed": failed,
        "total_detections": total_detections
    }


def process_batch_task(task_ids: List[int]):
    """后台处理批量任务（检测并匹配）"""
    from main import detect_match_service
    from database import SessionLocal
    from models.match_result import MatchResult

    detector = detect_match_service.detection_service.detector
    matcher = detect_match_service.match_service.matcher

    if detector is None or not detector.is_ready():
        return

    db = SessionLocal()
    try:
        for task_id in task_ids:
            try:
                task = db.query(Task).filter(Task.id == task_id).first()

                if not task or task.status == "detected":
                    continue

                with open(task.image_path, 'rb') as f:
                    image = process_uploaded_image(f.read())

                result = detector.detect_single_image(image, return_cropped=True, return_plot=True)

                boxes = result.get("detections", [])
                boxes = filter_small_boxes(
                    boxes,
                    image.size,
                    min_area_ratio=config.model.MIN_AREA_RATIO,
                    min_pixel_area=config.model.MIN_PIXEL_AREA
                )

                detected_boxes = []
                match_results = []
                sku_matcher_enabled = matcher is not None and matcher.is_ready()

                if sku_matcher_enabled and boxes:
                    features = []
                    for box in boxes:
                        cropped = crop_box(image, box.get("bbox", []))
                        if cropped:
                            resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                            feat = matcher.extract_feature(resized)
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
                            mr = matcher.match_sku(feat, threshold=0.85)
                            match_results.append({
                                'sku_id': mr.sku_id,
                                'sku_name': mr.sku_name,
                                'similarity': mr.similarity,
                                'status': mr.status,
                                'top5_labels': mr.top5_labels if mr.top5_labels else []
                            })
                else:
                    match_results = [None] * len(boxes)

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.get("bbox", [])
                    cropped = image.crop((x1, y1, x2, y2))

                    crops_dir = config.paths.TASKS_DIR / f"task_{task.id}" / "crops"
                    crops_dir.mkdir(exist_ok=True)
                    crop_path = crops_dir / f"box_{idx}.jpg"
                    cropped.save(crop_path)

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

                    mr = match_results[idx] if idx < len(match_results) else None
                    if mr:
                        top5_data = []
                        if mr.get('top5_labels'):
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
                            sku_id=mr.get('sku_id'),
                            sku_name=mr.get('sku_name'),
                            similarity=mr.get('similarity'),
                            status=mr.get('status', 'unmatched'),
                            top1_sku_id=mr.get('sku_id'),
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

                matched_count = sum(1 for mr in match_results if mr and mr.get('status') == 'matched')
                unmatched_count = sum(1 for mr in match_results if mr is None or mr.get('status') != 'matched')

                task_dir = config.paths.TASKS_DIR / f"task_{task.id}"
                task_dir.mkdir(exist_ok=True)
                
                try:
                    plot_image, _ = draw_detection_result(image, boxes, match_results)
                    plot_path = task_dir / "detection_result.jpg"
                    plot_image.save(plot_path, format='JPEG')
                    task.vis_image = str(plot_path)
                except Exception as e:
                    print(f"生成可视化结果失败: {e}")
                task.box_count = len(detected_boxes)
                task.matched_count = matched_count
                task.unmatched_count = unmatched_count
                task.status = "detected"
                task.completed_at = datetime.utcnow()

                db.commit()

            except Exception as e:
                print(f"处理任务 {task_id} 失败: {e}")
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    db.commit()
    finally:
        db.close()


@router.post("/batch", response_model=BatchTaskResponse)
async def create_batch_task(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """创建批量检测任务（仅检测，不匹配）"""
    valid_files = []
    for file in files:
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            valid_files.append(file)

    if not valid_files:
        raise HTTPException(status_code=400, detail="没有有效的图片文件")

    upload_dir = get_upload_dir()
    task_ids = []

    for file in valid_files:
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{file.filename}"
        file_path = upload_dir / filename

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        task = Task(
            task_name=file.filename,
            image_name=file.filename,
            image_path=str(file_path),
            status="pending",
            created_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_ids.append(task.id)

    background_tasks.add_task(process_batch_task, task_ids)

    return BatchTaskResponse(
        success=True,
        task_ids=task_ids,
        total_count=len(task_ids),
        message=f"已创建 {len(task_ids)} 个任务，正在后台检测中"
    )


@router.get("/batch/{task_ids}")
async def get_batch_task_status(
    task_ids: str,
    db: Session = Depends(get_db)
):
    """获取批量任务状态"""
    try:
        id_list = [int(id.strip()) for id in task_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的任务ID格式")

    tasks = db.query(Task).filter(Task.id.in_(id_list)).all()

    completed_count = sum(1 for t in tasks if t.status == "completed")
    failed_count = sum(1 for t in tasks if t.status == "failed")
    pending_count = sum(1 for t in tasks if t.status == "pending")
    detected_count = sum(1 for t in tasks if t.status == "detected")

    total_boxes = sum(t.box_count or 0 for t in tasks)
    total_matched = sum(t.matched_count or 0 for t in tasks)

    return {
        "success": True,
        "total_count": len(tasks),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "pending_count": pending_count,
        "detected_count": detected_count,
        "progress": (completed_count + failed_count) / len(tasks) * 100 if tasks else 0,
        "summary": {
            "total_boxes": total_boxes,
            "total_matched": total_matched,
            "success_rate": total_matched / total_boxes * 100 if total_boxes > 0 else 0
        },
        "tasks": [
            {
                "id": t.id,
                "image_name": t.image_name,
                "status": t.status,
                "box_count": t.box_count,
                "matched_count": t.matched_count,
                "unmatched_count": t.unmatched_count,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "result": t.result
            } for t in tasks
        ]
    }


@router.get("/{task_id}/export")
async def export_task_result(
    task_id: int,
    format: str = Query("json", enum=["json", "csv"]),
    include_images: bool = Query(False),
    db: Session = Depends(get_db)
):
    """导出任务检测和匹配结果"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    detection_boxes = db.query(DetectionBox).filter(
        DetectionBox.task_id == task_id
    ).order_by(DetectionBox.box_index).all()

    box_ids = [box.id for box in detection_boxes]
    match_results = db.query(MatchResult).filter(
        MatchResult.box_id.in_(box_ids)
    ).all()
    
    match_map = {mr.box_id: mr for mr in match_results}

    export_data = {
        "task_id": task.id,
        "image_name": task.image_name,
        "status": task.status,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "boxes": []
    }

    seen_box_indices = set()
    
    for db_box in detection_boxes:
        if db_box.box_index in seen_box_indices:
            continue
        seen_box_indices.add(db_box.box_index)

        box_item = {
            "box_id": f"box_{db_box.box_index}",
            "bbox": [db_box.bbox_x1, db_box.bbox_y1, db_box.bbox_x2, db_box.bbox_y2],
            "confidence": db_box.confidence,
            "class_name": db_box.class_name,
            "status": db_box.status,
            "is_audited": db_box.is_audited
        }

        if include_images and db_box.path:
            box_item["crop_path"] = db_box.path

        mr = match_map.get(db_box.id)
        if mr:
            box_item["sku_id"] = mr.sku_id
            box_item["similarity"] = mr.similarity
            box_item["match_status"] = mr.status
            box_item["top1_sku_id"] = mr.top1_sku_id

        export_data["boxes"].append(box_item)

    export_data["box_count"] = len(export_data["boxes"])
    export_data["matched_count"] = sum(1 for box in export_data["boxes"] if box.get("match_status") == "matched")
    export_data["unmatched_count"] = export_data["box_count"] - export_data["matched_count"]

    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(content=export_data)
    elif format == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "任务ID", "图片名称", "箱体编号", "检测置信度", "类别",
            "SKU编码", "SKU商品名称", "相似度", "匹配状态",
            "已审核", "人工修正", "修正时间", "坐标信息"
        ])
        
        for idx, box in enumerate(export_data["boxes"], 1):
            match_status_map = {
                "matched": "已匹配",
                "unmatched": "未匹配",
                "low_conf": "低置信"
            }
            match_status = match_status_map.get(box.get("match_status", ""), box.get("match_status", ""))
            
            bbox_str = f"[{box['bbox'][0]}, {box['bbox'][1]}, {box['bbox'][2]}, {box['bbox'][3]}]"
            
            writer.writerow([
                export_data["task_id"],
                export_data["image_name"],
                box["box_id"].replace("box_", "箱体"),
                f"{box['confidence']:.4f}" if box.get("confidence") else "",
                box.get("class_name", ""),
                box.get("sku_id", ""),
                box.get("sku_name", ""),
                f"{box['similarity']:.4f}" if box.get("similarity") else "",
                match_status,
                "是" if box.get("is_audited") else "否",
                "是" if box.get("is_manual_override") else "否",
                box.get("override_at", "")[:19] if box.get("override_at") else "",
                bbox_str
            ])
        
        output.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=task_{task_id}_export.csv"}
        )


@router.post("/batch/export")
async def export_batch_tasks(
    task_ids: List[int],
    format: str = Query("json", enum=["json", "csv"]),
    include_images: bool = Query(False),
    db: Session = Depends(get_db)
):
    """批量导出多个任务的检测和匹配结果"""
    if not task_ids:
        raise HTTPException(status_code=400, detail="请选择要导出的任务")

    tasks = db.query(Task).filter(Task.id.in_(task_ids)).all()
    
    if len(tasks) != len(task_ids):
        raise HTTPException(status_code=404, detail="部分任务不存在")

    export_data = {
        "tasks": [],
        "total_tasks": len(tasks),
        "exported_at": datetime.utcnow().isoformat()
    }

    for task in tasks:
        detection_boxes = db.query(DetectionBox).filter(
            DetectionBox.task_id == task.id
        ).order_by(DetectionBox.box_index).all()

        box_ids = [box.id for box in detection_boxes]
        match_results = db.query(MatchResult).filter(
            MatchResult.box_id.in_(box_ids)
        ).all()
        
        match_map = {mr.box_id: mr for mr in match_results}

        task_data = {
            "task_id": task.id,
            "image_name": task.image_name,
            "status": task.status,
            "image_path": task.image_path,
            "vis_image": task.vis_image,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "box_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "boxes": []
        }

        seen_box_indices = set()
        
        for db_box in detection_boxes:
            if db_box.box_index in seen_box_indices:
                continue
            seen_box_indices.add(db_box.box_index)

            box_item = {
                "box_id": f"box_{db_box.box_index}",
                "bbox": [db_box.bbox_x1, db_box.bbox_y1, db_box.bbox_x2, db_box.bbox_y2],
                "confidence": db_box.confidence,
                "class_name": db_box.class_name,
                "status": db_box.status,
                "is_audited": db_box.is_audited,
                "path": db_box.path
            }

            if include_images and db_box.path:
                try:
                    with open(db_box.path, 'rb') as f:
                        import base64
                        box_item["crop_base64"] = base64.b64encode(f.read()).decode('utf-8')
                except Exception:
                    pass

            mr = match_map.get(db_box.id)
            if mr:
                box_item["sku_id"] = mr.sku_id
                box_item["sku_name"] = mr.sku_name
                box_item["similarity"] = mr.similarity
                box_item["match_status"] = mr.status
                box_item["top1_sku_id"] = mr.top1_sku_id
                box_item["is_manual_override"] = mr.is_manual_override
                box_item["override_at"] = mr.override_at.isoformat() if mr.override_at else None

            task_data["boxes"].append(box_item)

        task_data["box_count"] = len(task_data["boxes"])
        task_data["matched_count"] = sum(1 for box in task_data["boxes"] if box.get("match_status") == "matched")
        task_data["unmatched_count"] = task_data["box_count"] - task_data["matched_count"]
        
        export_data["tasks"].append(task_data)

    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(content=export_data)
    elif format == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "任务ID", "图片名称", "任务状态", "创建时间",
            "箱体编号", "检测置信度(%)", "类别",
            "SKU编码", "SKU商品名称", "相似度", "匹配状态",
            "已审核", "人工修正", "修正时间", "坐标信息"
        ])
        
        for task in export_data["tasks"]:
            task_status_map = {
                "pending": "进行中",
                "detected": "已检测",
                "completed": "已完成",
                "failed": "失败"
            }
            task_status = task_status_map.get(task["status"], task["status"])
            
            for box in task["boxes"]:
                match_status_map = {
                    "matched": "已匹配",
                    "unmatched": "未匹配",
                    "low_conf": "低置信"
                }
                match_status = match_status_map.get(box.get("match_status", ""), box.get("match_status", ""))
                
                bbox_str = f"[{box['bbox'][0]}, {box['bbox'][1]}, {box['bbox'][2]}, {box['bbox'][3]}]"
                
                writer.writerow([
                    task["task_id"],
                    task["image_name"],
                    task_status,
                    task["created_at"][:19] if task["created_at"] else "",
                    box["box_id"].replace("box_", "箱体"),
                    f"{box['confidence']:.4f}" if box.get("confidence") else "",
                    box.get("class_name", ""),
                    box.get("sku_id", ""),
                    box.get("sku_name", ""),
                    f"{box['similarity']:.4f}" if box.get("similarity") else "",
                    match_status,
                    "是" if box.get("is_audited") else "否",
                    "是" if box.get("is_manual_override") else "否",
                    box.get("override_at", "")[:19] if box.get("override_at") else "",
                    bbox_str
                ])
        
        output.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_export_{len(tasks)}_tasks.csv"}
        )
