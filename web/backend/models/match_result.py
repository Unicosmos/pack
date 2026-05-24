"""
匹配结果模型
规范化存储每个检测框的SKU匹配结果
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from datetime import datetime
from database import Base


class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, index=True)
    box_id = Column(Integer, ForeignKey("detection_boxes.id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)

    sku_id = Column(String(50), nullable=True)
    sku_name = Column(String(200), nullable=True)
    similarity = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)

    top1_sku_id = Column(String(50), nullable=True)
    top5_candidates = Column(Text, nullable=True)

    is_manual_override = Column(Boolean, default=False)
    override_at = Column(DateTime, nullable=True)
    override_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MatchResult {self.id}: box={self.box_id}, sku={self.sku_id}, status={self.status}>"

    def to_dict(self):
        import json
        return {
            "box_id": str(self.box_id),
            "sku_id": self.sku_id,
            "sku_name": self.sku_name,
            "similarity": self.similarity,
            "status": self.status,
            "top1_sku_id": self.top1_sku_id,
            "top5_candidates": json.loads(self.top5_candidates) if self.top5_candidates else [],
            "is_manual_override": self.is_manual_override,
            "override_at": self.override_at.isoformat() if self.override_at else None,
            "override_reason": self.override_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
