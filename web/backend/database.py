"""
数据库配置模块
使用SQLAlchemy + SQLite
路径从config.py获取，避免硬编码
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# 延迟初始化，避免循环依赖
engine = None
SessionLocal = None
Base = declarative_base()


def _init_engine():
    """延迟初始化引擎"""
    global engine, SessionLocal
    if engine is None:
        from config import config
        db_path = config.paths.DATA_DIR / "pack.db"
        db_path.parent.mkdir(exist_ok=True)
        SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"
        
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            connect_args={"check_same_thread": False}
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """获取数据库会话"""
    _init_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库"""
    _init_engine()
    from models import user, task, sku
    Base.metadata.create_all(bind=engine)