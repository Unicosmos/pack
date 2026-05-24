"""
数据迁移脚本 v1_to_v2
将 Task.result JSON 中存储的检测框和匹配结果迁移到独立表

使用方法:
    python -m migrations.v1_to_v2

注意: 迁移前请备份数据库！
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import config
from database import Base, SessionLocal
from models.task import Task
from models.sku import SKU
from models.detection_box import DetectionBox
from models.match_result import MatchResult


def get_db_session():
    """获取数据库会话"""
    db_path = config.paths.DATA_DIR / "pack.db"
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    return Session(bind=engine)


def migrate_detection_boxes(task: Task, db: Session) -> List[int]:
    """
    迁移单个任务的检测框到 DetectionBox 表

    Returns:
        List[int]: 创建的 DetectionBox 的 ID 列表
    """
    box_ids = []

    if not task.result or "detections" not in task.result:
        return box_ids

    detections = task.result.get("detections", {})
    boxes = detections.get("boxes", [])

    for idx, box_data in enumerate(boxes):
        bbox = box_data.get("bbox", [0, 0, 0, 0])

        detection_box = DetectionBox(
            task_id=task.id,
            box_index=idx,
            bbox_x1=bbox[0] if len(bbox) > 0 else 0,
            bbox_y1=bbox[1] if len(bbox) > 1 else 0,
            bbox_x2=bbox[2] if len(bbox) > 2 else 0,
            bbox_y2=bbox[3] if len(bbox) > 3 else 0,
            confidence=box_data.get("confidence", 0.0),
            class_id=box_data.get("class_id", 0),
            class_name=box_data.get("class_name", "box"),
            path=box_data.get("path") or box_data.get("crop_path"),
            status=box_data.get("status", "approved"),
            is_audited=box_data.get("is_audited", False)
        )
        db.add(detection_box)
        db.flush()
        box_ids.append(detection_box.id)

    return box_ids


def migrate_match_results(task_id: int, box_id: int, match_data: Dict[str, Any], db: Session):
    """迁移匹配结果到 MatchResult 表"""
    if not match_data:
        return

    top5_labels = match_data.get("top5_labels", []) or match_data.get("top5", [])
    top5_data = []
    for label in top5_labels:
        top5_data.append({
            "sku_id": label.get("sku_id", ""),
            "name": label.get("label", ""),
            "similarity": label.get("similarity", 0),
            "image_path": label.get("image_path", "")
        })

    match_result = MatchResult(
        box_id=box_id,
        task_id=task_id,
        sku_id=match_data.get("sku_id"),
        sku_name=match_data.get("sku_name"),
        similarity=match_data.get("similarity"),
        status=match_data.get("status", "unmatched"),
        top1_sku_id=match_data.get("sku_id"),
        top5_candidates=json.dumps(top5_data) if top5_data else None,
        is_manual_override=False,
    )
    db.add(match_result)


def migrate_task(task: Task, db: Session) -> bool:
    """
    迁移单个任务

    Returns:
        bool: 是否迁移成功
    """
    try:
        box_ids = migrate_detection_boxes(task, db)

        if not box_ids:
            return True

        matches = task.result.get("matches", {})
        for box_key, match_data in matches.items():
            if str(box_key).isdigit():
                idx = int(box_key)
            elif box_key.startswith("box_"):
                idx = int(box_key.split("_")[1])
            else:
                idx = -1

            if idx >= 0 and idx < len(box_ids):
                migrate_match_results(task.id, box_ids[idx], match_data, db)

        return True
    except Exception as e:
        print(f"迁移任务 {task.id} 失败: {e}")
        return False


def run_migration():
    """执行迁移"""
    print("=" * 60)
    print("数据迁移 v1 -> v2")
    print("=" * 60)
    print()

    db = get_db_session()

    try:
        tasks = db.query(Task).all()
        total = len(tasks)
        success = 0
        failed = 0

        print(f"找到 {total} 个任务需要迁移...")
        print()

        for i, task in enumerate(tasks, 1):
            if migrate_task(task, db):
                success += 1
                print(f"[{i}/{total}] 任务 {task.id} 迁移成功")
            else:
                failed += 1
                print(f"[{i}/{total}] 任务 {task.id} 迁移失败")

        db.commit()
        print()
        print("=" * 60)
        print(f"迁移完成！成功: {success}, 失败: {failed}")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"迁移过程中发生错误: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("警告: 此脚本将修改数据库！")
    print("请确保已备份数据库！")
    print()
    response = input("是否继续？(yes/no): ")
    if response.lower() == "yes":
        run_migration()
    else:
        print("已取消迁移")
