"""
导出相关API
支持任务检测结果的导出功能
"""

import csv
import json
from datetime import datetime
from io import StringIO
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models.task import Task
from models.detection_box import DetectionBox
from models.match_result import MatchResult

router = APIRouter(prefix="/api/tasks", tags=["任务管理"])


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
        "created_at": task.created_at.isoformat() + 'Z' if task.created_at else None,
        "completed_at": task.completed_at.isoformat() + 'Z' if task.completed_at else None,
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
        return JSONResponse(content=export_data)
    elif format == "csv":
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
        "exported_at": datetime.utcnow().isoformat() + 'Z'
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
            "created_at": task.created_at.isoformat() + 'Z' if task.created_at else None,
            "completed_at": task.completed_at.isoformat() + 'Z' if task.completed_at else None,
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
        return JSONResponse(content=export_data)
    elif format == "csv":
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
                    f"{box['confidence'] * 100:.1f}" if box.get("confidence") else "",
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
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=batch_tasks_export.csv"}
        )