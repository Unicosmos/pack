"""
任务数据访问层 - 封装任务相关的数据库操作
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from sqlalchemy.orm import Session

from models.task import Task
from config import config


class TaskRepository:
    """任务数据访问类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, task_name: str, image_name: str, image_content: bytes) -> Task:
        """
        创建新任务
        
        Args:
            task_name: 任务名称
            image_name: 图片名称
            image_content: 图片内容
            
        Returns:
            创建的任务对象
        """
        unique_id = str(uuid4())[:8]
        filename = f"{unique_id}_{image_name}"
        upload_dir = config.paths.DATA_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(image_content)
        
        task = Task(
            task_name=task_name,
            image_name=image_name,
            image_path=str(file_path),
            status="uploaded",
            result={},
            box_count=0,
            created_at=datetime.utcnow(),
            completed_at=None
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def get_by_id(self, task_id: int) -> Optional[Task]:
        """根据ID获取任务"""
        return self.db.query(Task).filter(Task.id == task_id).first()
    
    def list(self, page: int = 1, page_size: int = 10, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        分页获取任务列表
        
        Args:
            page: 页码
            page_size: 每页数量
            status_filter: 状态过滤
            
        Returns:
            任务列表和总数
        """
        query = self.db.query(Task)
        
        if status_filter:
            query = query.filter(Task.status == status_filter)
        
        total = query.count()
        offset = (page - 1) * page_size
        tasks = query.offset(offset).limit(page_size).all()
        
        return {
            "tasks": tasks,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    def update(self, task_id: int, data: Dict[str, Any]) -> Optional[Task]:
        """
        更新任务
        
        Args:
            task_id: 任务ID
            data: 更新数据
            
        Returns:
            更新后的任务对象
        """
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        for key, value in data.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def delete(self, task_id: int) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否删除成功
        """
        task = self.get_by_id(task_id)
        if not task:
            return False
        
        # 删除关联文件
        try:
            if task.image_path and Path(task.image_path).exists():
                Path(task.image_path).unlink()
        except Exception:
            pass
        
        self.db.delete(task)
        self.db.commit()
        
        return True
    
    def update_result(self, task_id: int, result: Dict[str, Any]) -> Optional[Task]:
        """
        更新任务结果
        
        Args:
            task_id: 任务ID
            result: 检测/匹配结果
            
        Returns:
            更新后的任务对象
        """
        return self.update(task_id, {
            "result": result,
            "completed_at": datetime.utcnow()
        })
    
    def get_stats(self) -> Dict[str, int]:
        """获取任务统计"""
        total = self.db.query(Task).count()
        completed = self.db.query(Task).filter(Task.status == "completed").count()
        detected = self.db.query(Task).filter(Task.status == "detected").count()
        
        return {
            "total": total,
            "completed": completed,
            "detected": detected,
            "pending": total - completed - detected
        }
