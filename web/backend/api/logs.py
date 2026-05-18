"""
操作日志API
提供操作日志的查询接口
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from models.operation_log import OperationLog
from schemas.schemas import OperationLogListResponse, OperationLogSchema

router = APIRouter(prefix="/api/logs", tags=["操作日志"])


@router.get("", response_model=OperationLogListResponse)
async def list_logs(
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    entity_id: Optional[int] = Query(None, description="实体ID过滤"),
    action: Optional[str] = Query(None, description="操作类型过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    获取操作日志列表

    支持按实体类型、实体ID、操作类型过滤
    """
    query = db.query(OperationLog)

    if entity_type:
        query = query.filter(OperationLog.entity_type == entity_type)
    if entity_id:
        query = query.filter(OperationLog.entity_id == entity_id)
    if action:
        query = query.filter(OperationLog.action == action)

    total = query.count()

    logs = query.order_by(OperationLog.operated_at.desc()) \
        .offset((page - 1) * page_size) \
        .limit(page_size) \
        .all()

    return OperationLogListResponse(
        success=True,
        logs=[OperationLogSchema.model_validate(log.to_dict()) for log in logs],
        total=total
    )


@router.get("/{log_id}", response_model=OperationLogSchema)
async def get_log(
    log_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个操作日志详情
    """
    log = db.query(OperationLog).filter(OperationLog.id == log_id).first()
    if not log:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="日志不存在")

    return OperationLogSchema.model_validate(log.to_dict())
