"""
任务管理API（基础CRUD）
支持任务的创建、查询、更新、删除
"""

import os
import shutil
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from pathlib import Path

from database import get_db
from config import config
from core.utils.image_utils import process_uploaded_image
from core.visualizer import draw_detection_result
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult
from schemas.schemas import TaskResponse, TaskUpdate, BatchTaskResponse

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


def process_batch_task(task_ids: List[int]):
    """后台处理批量任务（检测并匹配）"""
    from main import detect_match_service
    from database import SessionLocal

    if not detect_match_service.is_detection_ready():
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

                detect_result = detect_match_service.detect(image)
                boxes = detect_result.get("boxes", [])
                plot_image = detect_result.get("plot_image")

                detected_boxes = []
                sku_matcher_enabled = detect_match_service.is_match_ready()

                if sku_matcher_enabled and boxes:
                    match_results, matched_count, unmatched_count = detect_match_service.match(image, boxes)
                else:
                    match_results = [None] * len(boxes)
                    matched_count = 0
                    unmatched_count = len(boxes)

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

                task_dir = config.paths.TASKS_DIR / f"task_{task.id}"
                task_dir.mkdir(exist_ok=True)

                try:
                    if plot_image:
                        plot_path = task_dir / "detection_result.jpg"
                        plot_image.save(plot_path, format='JPEG')
                        task.vis_image = str(plot_path)
                    else:
                        plot_img, _ = draw_detection_result(image, boxes, match_results)
                        plot_path = task_dir / "detection_result.jpg"
                        plot_img.save(plot_path, format='JPEG')
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

    upload_dir = config.paths.DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
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


class TaskListResponse(BaseModel):
    success: bool
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[str] = None,
    time_filter: Optional[str] = Query(None, description="时间筛选: today/week/month"),
    start_time: Optional[str] = Query(None, description="自定义开始时间 ISO格式"),
    end_time: Optional[str] = Query(None, description="自定义结束时间 ISO格式"),
    db: Session = Depends(get_db)
):
    """获取任务列表"""
    query = db.query(Task)

    if status_filter:
        query = query.filter(Task.status == status_filter)

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            query = query.filter(Task.created_at >= start_dt)
        except ValueError:
            pass

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
            query = query.filter(Task.created_at <= end_dt)
        except ValueError:
            pass

    if not start_time and not end_time:
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        week_start = today_start - timedelta(days=today.weekday())
        month_start = today_start.replace(day=1)

        if time_filter == "today":
            query = query.filter(Task.created_at >= today_start)
        elif time_filter == "week":
            query = query.filter(Task.created_at >= week_start)
        elif time_filter == "month":
            query = query.filter(Task.created_at >= month_start)

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


@router.post("/batch-delete")
async def batch_delete_tasks(
    task_ids: List[int],
    db: Session = Depends(get_db)
):
    """批量删除任务"""
    if not task_ids:
        raise HTTPException(status_code=400, detail="请提供要删除的任务ID")

    tasks = db.query(Task).filter(Task.id.in_(task_ids)).all()

    if not tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    deleted_ids = []
    for task in tasks:
        db.query(MatchResult).filter(MatchResult.task_id == task.id).delete(synchronize_session=False)
        db.query(DetectionBox).filter(DetectionBox.task_id == task.id).delete()

        if os.path.exists(task.image_path):
            os.remove(task.image_path)

        task_dir = config.paths.TASKS_DIR / f"task_{task.id}"
        if task_dir.exists():
            shutil.rmtree(task_dir)

        db.delete(task)
        deleted_ids.append(task.id)

    db.commit()

    return {"success": True, "message": f"已删除 {len(deleted_ids)} 个任务", "deleted_ids": deleted_ids}


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


@router.get("/stats/summary")
async def get_task_stats(
    time_filter: Optional[str] = Query(None, description="时间筛选: today/week/month"),
    start_time: Optional[str] = Query(None, description="自定义开始时间 ISO格式"),
    end_time: Optional[str] = Query(None, description="自定义结束时间 ISO格式"),
    db: Session = Depends(get_db)
):
    """获取任务统计（支持时间筛选）"""
    query = db.query(Task)

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            query = query.filter(Task.created_at >= start_dt)
        except ValueError:
            pass

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
            query = query.filter(Task.created_at <= end_dt)
        except ValueError:
            pass

    if not start_time and not end_time:
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        week_start = today_start - timedelta(days=today.weekday())
        month_start = today_start.replace(day=1)

        if time_filter == "today":
            query = query.filter(Task.created_at >= today_start)
        elif time_filter == "week":
            query = query.filter(Task.created_at >= week_start)
        elif time_filter == "month":
            query = query.filter(Task.created_at >= month_start)

    total = query.count()
    completed = query.filter(Task.status == "completed").count()
    pending = query.filter(Task.status == "pending").count()
    detected = query.filter(Task.status == "detected").count()
    failed = query.filter(Task.status == "failed").count()

    total_boxes = query.with_entities(func.sum(Task.box_count)).scalar() or 0
    matched_boxes = query.with_entities(func.sum(Task.matched_count)).scalar() or 0
    unmatched_boxes = query.with_entities(func.sum(Task.unmatched_count)).scalar() or 0
    match_rate = (matched_boxes / total_boxes * 100) if total_boxes > 0 else 0

    filtered_task_ids = [t.id for t in query.all()]

    sku_category_count = 0
    sku_distribution = []
    if filtered_task_ids:
        match_query = db.query(MatchResult).filter(
            MatchResult.task_id.in_(filtered_task_ids),
            MatchResult.sku_id.isnot(None)
        )
        sku_category_count = match_query.with_entities(MatchResult.sku_id).distinct().count()

        sku_dist_query = db.query(
            MatchResult.sku_id,
            MatchResult.sku_name,
            func.count(MatchResult.id).label('count')
        ).filter(
            MatchResult.task_id.in_(filtered_task_ids),
            MatchResult.sku_id.isnot(None),
            MatchResult.status == "matched"
        ).group_by(MatchResult.sku_id, MatchResult.sku_name).order_by(
            func.count(MatchResult.id).desc()
        ).limit(10).all()

        sku_distribution = []
        for sku_id, sku_name, count in sku_dist_query:
            sku_distribution.append({
                "sku_id": sku_id,
                "sku_name": sku_name or "未知",
                "count": count
            })

    if not start_time and not end_time:
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        week_start = today_start - timedelta(days=today.weekday())
        today_tasks = db.query(Task).filter(Task.created_at >= today_start).count()
        week_tasks = db.query(Task).filter(Task.created_at >= week_start).count()
    else:
        today_tasks = 0
        week_tasks = 0

    return {
        "success": True,
        "total": total,
        "completed": completed,
        "pending": pending,
        "detected": detected,
        "failed": failed,
        "total_boxes": total_boxes,
        "today_tasks": today_tasks,
        "week_tasks": week_tasks,
        "warehouse": {
            "total_boxes": total_boxes,
            "matched_boxes": matched_boxes,
            "unmatched_boxes": unmatched_boxes,
            "match_rate": round(match_rate, 2)
        },
        "sku": {
            "category_count": sku_category_count,
            "distribution": sku_distribution
        }
    }


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
                "created_at": t.created_at.isoformat() + 'Z' if t.created_at else None,
                "result": t.result
            } for t in tasks
        ]
    }