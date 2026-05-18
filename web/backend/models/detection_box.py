"""
检测框模型
规范化存储检测结果中的每个框
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, JSON
from datetime import datetime
from database import Base


class DetectionBox(Base):
    __tablename__ = "detection_boxes"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    box_index = Column(Integer, nullable=False)

    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)

    confidence = Column(Float, nullable=False)
    class_id = Column(Integer, default=0)
    class_name = Column(String(50), default="box")

    path = Column(String(1000), nullable=True)

    status = Column(String(20), default="pending")
    is_audited = Column(Boolean, default=False)
    reviewed_at = Column(DateTime, nullable=True)

    extra_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DetectionBox {self.id}: task={self.task_id}, box={self.box_index}, conf={self.confidence:.2f}>"

    def to_dict(self):
        return {
            "box_id": f"box_{self.box_index}",
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "path": self.path,
            "status": self.status,
            "is_audited": self.is_audited,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
