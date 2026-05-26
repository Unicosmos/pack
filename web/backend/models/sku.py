"""
SKU模型
支持完整的CRUD操作
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from datetime import datetime
from database import Base


class SKU(Base):
    __tablename__ = "skus"

    id = Column(Integer, primary_key=True, index=True)
    sku_id = Column(String(50), unique=True, index=True, nullable=False, comment="SKU编号")
    sku_name = Column(String(200), nullable=False, comment="SKU名称")
    description = Column(Text, nullable=True, comment="SKU描述")
    category = Column(String(100), nullable=True, index=True, comment="分类")
    image_count = Column(Integer, default=0, comment="图片数量")
    tags = Column(String(500), nullable=True, comment="标签，逗号分隔")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    created_by = Column(Integer, nullable=True, comment="创建人ID")
    is_deleted = Column(Boolean, default=False, index=True, comment="禁用标记: True-已禁用, False-启用")

    def __repr__(self):
        return f"<SKU {self.sku_id}: {self.sku_name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "sku_id": self.sku_id,
            "sku_name": self.sku_name,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "image_count": self.image_count,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
