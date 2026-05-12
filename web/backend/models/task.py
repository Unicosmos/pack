"""
任务模型
存储检测任务记录
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from datetime import datetime
from database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    task_name = Column(String(200), nullable=True)
    image_name = Column(String(500), nullable=False)
    image_path = Column(String(1000), nullable=False)
    status = Column(String(20), default="pending")
    result = Column(JSON, nullable=True)
    box_count = Column(Integer, default=0)
    matched_count = Column(Integer, default=0)
    unmatched_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Task {self.id}: {self.image_name}>"
