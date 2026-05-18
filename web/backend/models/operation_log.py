"""
操作日志模型
记录所有人工干预行为，确保可审计性
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from database import Base


class OperationLog(Base):
    __tablename__ = "operation_logs"

    id = Column(Integer, primary_key=True, index=True)

    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(Integer, nullable=False, index=True)

    action = Column(String(50), nullable=False)

    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)

    operator_ip = Column(String(50), nullable=True)

    operated_at = Column(DateTime, default=datetime.utcnow, index=True)

    remark = Column(Text, nullable=True)

    def __repr__(self):
        return f"<OperationLog {self.id}: {self.entity_type}/{self.entity_id} - {self.action}>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "old_value": json.loads(self.old_value) if self.old_value else None,
            "new_value": json.loads(self.new_value) if self.new_value else None,
            "operator_ip": self.operator_ip,
            "operated_at": self.operated_at.isoformat() if self.operated_at else None,
            "remark": self.remark
        }


def log_operation(db, entity_type: str, entity_id: int, action: str,
                  old_value: dict = None, new_value: dict = None,
                  operator_ip: str = None, remark: str = None):
    """
    记录操作日志的便捷函数

    Args:
        db: 数据库会话
        entity_type: 实体类型（如 'task', 'box', 'match'）
        entity_id: 实体ID
        action: 操作类型（如 'create', 'update', 'delete', 'review', 'override'）
        old_value: 修改前的值（dict）
        new_value: 修改后的值（dict）
        operator_ip: 操作者IP
        remark: 备注
    """
    import json
    log = OperationLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        old_value=json.dumps(old_value) if old_value else None,
        new_value=json.dumps(new_value) if new_value else None,
        operator_ip=operator_ip,
        remark=remark
    )
    db.add(log)
    return log
