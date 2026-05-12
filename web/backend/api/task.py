"""
任务管理API
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from auth import get_current_user_required
from database import get_db
from models.user import User
from models.task import Task

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


def get_upload_dir() -> Path:
    """获取上传目录（从config获取）"""
    from config import config
    upload_dir = config.paths.DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


class TaskResponse(BaseModel):
    id: int
    image_name: str
    status: str
    box_count: int
    matched_count: int
    unmatched_count: int
    result: Optional[dict]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class TaskListResponse(BaseModel):
    success: bool
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    result: Optional[dict] = None
    box_count: Optional[int] = None
    matched_count: Optional[int] = None
    unmatched_count: Optional[int] = None
    error_message: Optional[str] = None


@router.post("/upload", response_model=TaskResponse)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """上传图片并创建任务"""
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="只支持图片格式：jpg, png, bmp")

    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{file.filename}"
    upload_dir = get_upload_dir()
    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    task = Task(
        user_id=current_user.id,
        task_name=file.filename,
        image_name=file.filename,
        image_path=str(file_path),
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    return task


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """获取任务列表"""
    query = db.query(Task).filter(Task.user_id == current_user.id)

    if status_filter:
        query = query.filter(Task.status == status_filter)

    total = query.count()
    tasks = query.order_by(Task.created_at.desc()) \
        .offset((page - 1) * page_size) \
        .limit(page_size) \
        .all()

    return TaskListResponse(
        success=True,
        tasks=tasks,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """获取任务详情"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return task


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    update_data: TaskUpdate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """更新任务"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()

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
    if update_data.error_message is not None:
        task.error_message = update_data.error_message

    if update_data.status == "completed":
        task.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(task)
    return task


@router.delete("/{task_id}")
async def delete_task(
    task_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """删除任务"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if os.path.exists(task.image_path):
        os.remove(task.image_path)

    db.delete(task)
    db.commit()

    return {"success": True, "message": "任务已删除"}


@router.get("/stats/summary")
async def get_task_stats(
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """获取任务统计"""
    total = db.query(Task).filter(Task.user_id == current_user.id).count()
    completed = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.status == "completed"
    ).count()
    pending = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.status == "pending"
    ).count()
    failed = db.query(Task).filter(
        Task.user_id == current_user.id,
        Task.status == "failed"
    ).count()

    total_detections = db.query(Task).filter(
        Task.user_id == current_user.id
    ).with_entities(func.sum(Task.box_count)).scalar() or 0

    return {
        "success": True,
        "total": total,
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "total_detections": total_detections
    }