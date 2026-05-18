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
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from database import get_db
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from models.operation_log import log_operation
from schemas.schemas import (
    TaskResponse,
    TaskUpdate,
    ReviewUpdate,
    ReviewResponse,
    DetectedBox,
)

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])

executor = ThreadPoolExecutor(max_workers=4)


def get_upload_dir() -> Path:
    """获取上传目录（从config获取）"""
    from config import config
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
        detection_status=task.detection_status,
        review_status=task.review_status,
        box_count=task.box_count,
        matched_count=task.matched_count,
        unmatched_count=task.unmatched_count,
        result=task.result,
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

    unique_id = str(uuid.uuid4())[:8]
    # 安全处理文件名
    safe_filename = file.filename.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f"{unique_id}_{safe_filename}"
    upload_dir = get_upload_dir()
    file_path = upload_dir / filename
    
    # 确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败：{str(e)}")

    task = Task(
        task_name=file.filename,
        image_name=file.filename,
        image_path=str(file_path),
        status="pending",
        detection_status="pending",
        review_status="pending",
        created_at=datetime.utcnow()
    )
    db.add(task)
    db.commit()
    db.refresh(task)

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
    if update_data.result is not None:
        task.result = update_data.result
    if update_data.box_count is not None:
        task.box_count = update_data.box_count
    if update_data.matched_count is not None:
        task.matched_count = update_data.matched_count
    if update_data.unmatched_count is not None:
        task.unmatched_count = update_data.unmatched_count
    if update_data.detection_status is not None:
        task.detection_status = update_data.detection_status
    if update_data.review_status is not None:
        task.review_status = update_data.review_status
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
    from main import detector, matcher
    from core.utils.image_utils import process_uploaded_image, filter_small_boxes, image_to_base64, crop_box, resize_with_padding
    from schemas.schemas import BoxInfo, MatchInfo, TopLabel
    from core.visualizer import draw_detection_result
    from config import config

    if detector is None or not detector.is_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.detection_status == "detected" and task.review_status != "pending":
        return task_to_response(task)

    try:
        with open(task.image_path, 'rb') as f:
            image = process_uploaded_image(f.read())

        result = detector.detect_single_image(image, return_cropped=True, return_plot=True)

        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)

        if not boxes:
            task.detection_status = "detected"
            task.review_status = "reviewed"
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
                            'similarity': 0.0,
                            'ratio': None,
                            'status': 'unmatched',
                            'top5_labels': []
                        })
                    else:
                        mr = matcher.match_sku(feat, threshold=match_threshold)
                        match_results.append({
                            'sku_id': mr.sku_id,
                            'similarity': mr.similarity,
                            'ratio': mr.ratio,
                            'status': mr.status,
                            'top5_labels': mr.top5_labels if mr.top5_labels else []
                        })
            except Exception as e:
                print(f"匹配失败: {e}")
                sku_matcher_enabled = False

        if not sku_matcher_enabled:
            match_results = [None] * len(boxes)

        detected_boxes = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.get("bbox", [])
            cropped = image.crop((x1, y1, x2, y2))
            crop_base64 = image_to_base64(cropped)
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
                path=None,
                status="approved",
                is_audited=False,
                extra_data={
                    "crop_base64": crop_base64,
                    "match_result": mr
                }
            )
            db.add(detection_box)

            detected_boxes.append({
                "box_id": str(idx),
                "bbox": [x1, y1, x2, y2],
                "confidence": box.get("confidence", 0.0),
                "class_id": box.get("class_id", 0),
                "class_name": box.get("class_name", "box"),
                "status": "approved",
                "is_audited": False,
                "crop_base64": crop_base64,
                "match_result": mr
            })

        matched_count = sum(1 for mr in match_results if mr and mr.get('status') == 'matched')
        unmatched_count = sum(1 for mr in match_results if mr is None or mr.get('status') == 'unmatched')

        task.result = {
            "detections": {
                "boxes": detected_boxes
            },
            "matches": match_results
        }
        task.box_count = len(detected_boxes)
        task.matched_count = matched_count
        task.unmatched_count = unmatched_count
        task.detection_status = "detected"
        task.review_status = "pending"
        task.status = "detected"
        task.completed_at = datetime.utcnow()

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
    """获取任务图片"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not os.path.exists(task.image_path):
        raise HTTPException(status_code=404, detail="图片不存在")

    from fastapi.responses import FileResponse
    return FileResponse(task.image_path)

@router.get("/{task_id}/detections")
async def get_task_detections(
    task_id: int,
    db: Session = Depends(get_db)
):
    """获取任务的检测结果"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not task.result or "detections" not in task.result:
        return {
            "success": True,
            "task_id": task_id,
            "boxes": []
        }

    return {
        "success": True,
        "task_id": task_id,
        "detection_status": task.detection_status,
        "review_status": task.review_status,
        "boxes": task.result.get("detections", {}).get("boxes", []),
        "image_with_boxes": task.result.get("image_with_boxes")
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

    if task.detection_status != "detected":
        raise HTTPException(status_code=400, detail="任务尚未完成检测")

    try:
        approved_count = 0
        rejected_count = 0
        deleted_count = 0

        detection_boxes = db.query(DetectionBox).filter(
            DetectionBox.task_id == task_id
        ).order_by(DetectionBox.box_index).all()

        # 创建box_id到审核数据的映射
        review_box_map = {box.get("box_id"): box for box in review_data.boxes}

        for idx, db_box in enumerate(detection_boxes):
            box_id_str = f"box_{idx}"
            
            if box_id_str in review_box_map:
                review_box = review_box_map[box_id_str]
                new_status = review_box.get("status", "approved")
                old_status = db_box.status

                # 保存自定义SKU
                if "custom_sku" in review_box:
                    db_box.extra_data = db_box.extra_data or {}
                    db_box.extra_data["custom_sku"] = review_box["custom_sku"]

                if old_status != new_status:
                    log_operation(
                        db=db,
                        entity_type="box",
                        entity_id=db_box.id,
                        action="review",
                        old_value={"status": old_status},
                        new_value={"status": new_status}
                    )

                db_box.status = new_status
                db_box.is_audited = True
                db_box.reviewed_at = datetime.utcnow()

                if new_status == "approved":
                    approved_count += 1
                elif new_status == "rejected":
                    rejected_count += 1
                elif new_status == "deleted":
                    deleted_count += 1

        # 更新task.result中的boxes，确保box_id格式统一
        if "detections" not in task.result:
            task.result["detections"] = {}
        
        # 保存完整的审核数据，包括自定义SKU
        task.result["detections"]["boxes"] = review_data.boxes
        task.review_status = "reviewed"
        task.box_count = approved_count

        log_operation(
            db=db,
            entity_type="task",
            entity_id=task_id,
            action="review",
            old_value={"review_status": "pending"},
            new_value={"review_status": "reviewed", "approved": approved_count, "rejected": rejected_count, "deleted": deleted_count}
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
    from main import matcher, detector
    from core.utils.image_utils import process_uploaded_image, crop_box, resize_with_padding

    if matcher is None or not matcher.is_ready():
        raise HTTPException(status_code=503, detail="SKU匹配器未加载")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.review_status != "reviewed":
        raise HTTPException(status_code=400, detail="任务尚未完成审核")

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
                    sku_id=None,
                    similarity=None,
                    status="unmatched",
                    top1_sku_id=None
                )
                db.add(match_result)
                matches[f"box_{db_box.box_index}"] = {"status": "unmatched", "sku_id": None, "similarity": None}
                unmatched_count += 1
                continue

            # 检查是否有自定义SKU
            custom_sku = None
            if db_box.extra_data and db_box.extra_data.get("custom_sku"):
                custom_sku = db_box.extra_data["custom_sku"]

            if custom_sku:
                # 如果有自定义SKU，直接使用
                match_result = MatchResult(
                    box_id=db_box.id,
                    sku_id=custom_sku,
                    similarity=1.0,
                    status="matched",
                    top1_sku_id=custom_sku,
                    top5_candidates=None
                )
                db.add(match_result)
                
                matches[f"box_{db_box.box_index}"] = {
                    "sku_id": custom_sku,
                    "similarity": 1.0,
                    "status": "matched"
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
                            "label": label.get("label", ""),
                            "similarity": label.get("similarity", 0),
                            "image_path": label.get("image_path", "")
                        })

                match_result = MatchResult(
                    box_id=db_box.id,
                    sku_id=result.sku_id,
                    similarity=result.similarity,
                    status=result.status,
                    top1_sku_id=result.sku_id,
                    top5_candidates=json.dumps(top5_data) if top5_data else None
                )
                db.add(match_result)

                matches[f"box_{db_box.box_index}"] = {
                    "sku_id": result.sku_id,
                    "similarity": result.similarity,
                    "status": result.status
                }

                if result.status == "matched":
                    matched_count += 1
                else:
                    unmatched_count += 1

        task.result["matches"] = matches
        task.matched_count = matched_count
        task.unmatched_count = unmatched_count
        task.review_status = "matched"
        task.status = "completed"
        task.completed_at = datetime.utcnow()

        log_operation(
            db=db,
            entity_type="task",
            entity_id=task_id,
            action="match",
            old_value={"review_status": "reviewed"},
            new_value={"review_status": "matched", "matched": matched_count, "unmatched": unmatched_count}
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

    if os.path.exists(task.image_path):
        os.remove(task.image_path)

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
    detected = db.query(Task).filter(Task.detection_status == "detected").count()
    reviewed = db.query(Task).filter(Task.review_status == "reviewed").count()
    failed = db.query(Task).filter(Task.status == "failed").count()

    total_detections = db.query(Task).with_entities(func.sum(Task.box_count)).scalar() or 0

    return {
        "success": True,
        "total": total,
        "completed": completed,
        "pending": pending,
        "detected": detected,
        "reviewed": reviewed,
        "failed": failed,
        "total_detections": total_detections
    }


def process_batch_task(task_ids: List[int]):
    """后台处理批量任务（仅检测，不匹配）"""
    from main import detector
    from database import SessionLocal
    from core.utils.image_utils import process_uploaded_image, filter_small_boxes, image_to_base64
    from config import config

    if detector is None or not detector.is_ready():
        return

    db = SessionLocal()
    try:
        for task_id in task_ids:
            try:
                task = db.query(Task).filter(Task.id == task_id).first()

                if not task or task.detection_status == "detected":
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
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.get("bbox", [])
                    cropped = image.crop((x1, y1, x2, y2))
                    crop_base64 = image_to_base64(cropped)

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
                        path=None,
                        status="approved",
                        is_audited=False,
                        extra_data={"crop_base64": crop_base64}
                    )
                    db.add(detection_box)

                    detected_boxes.append({
                        "box_id": str(idx),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": box.get("confidence", 0.0),
                        "class_id": box.get("class_id", 0),
                        "class_name": box.get("class_name", "box"),
                        "status": "approved",
                        "is_audited": False,
                        "crop_base64": crop_base64
                    })

                task.result = {
                    "detections": {"boxes": detected_boxes},
                    "matches": {},
                    "image_with_boxes": result.get("plot_base64")
                }
                task.box_count = len(detected_boxes)
                task.detection_status = "detected"
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
            detection_status="pending",
            review_status="pending",
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
    detected_count = sum(1 for t in tasks if t.detection_status == "detected" and t.review_status != "reviewed")

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
                "detection_status": t.detection_status,
                "review_status": t.review_status,
                "box_count": t.box_count,
                "matched_count": t.matched_count,
                "unmatched_count": t.unmatched_count,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "result": t.result
            } for t in tasks
        ]
    }
