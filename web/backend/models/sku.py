"""
SKU模型
存储SKU管理信息（简化版，核心数据仍在csv和npy文件中）
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from database import Base


class SKU(Base):
    __tablename__ = "skus"

    id = Column(Integer, primary_key=True, index=True)
    sku_id = Column(String(50), unique=True, index=True, nullable=False)
    sku_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)
    image_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SKU {self.sku_id}: {self.sku_name}>"
