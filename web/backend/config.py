"""
Pack Web 配置管理模块
集中管理所有配置参数
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    CONF_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    MIN_AREA_RATIO: float = 0.01
    MIN_PIXEL_AREA: int = 2500
    INPUT_SIZE: int = 224


@dataclass
class MatchConfig:
    MATCH_THRESHOLD: float = 0.85
    RATIO_THRESHOLD: float = 1.2
    FEATURE_DIM: int = 384
    TOP_K: int = 5


@dataclass
class PathConfig:
    BASE_DIR: Path
    CORE_DIR: Path
    DATA_DIR: Path
    MODEL_PATH: Path
    SKU_DIR: Path
    SKU_FEATURES: Path
    SKU_INDEX: Path
    SKU_IMAGES_DIR: Path
    SKU_MODEL_PATH: Optional[Path]
    ULTRALYTICS_DIR: Path
    YOLO_CONFIG_DIR: Path


class Config:
    """单例配置类"""

    _instance: Optional['Config'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._load_paths()
        self.model = ModelConfig()
        self.match = MatchConfig()

    def _load_paths(self):
        """加载路径配置"""
        backend_dir = Path(__file__).parent
        base_dir = backend_dir.parent.parent

        data_dir = base_dir / "data"
        models_dir = data_dir / "models"
        sku_library_dir = data_dir / "sku_library"

        sku_model_path = models_dir / "sku" / "vits16_dino.pth"
        if not sku_model_path.exists():
            sku_model_path = None

        self.paths = PathConfig(
            BASE_DIR=base_dir,
            CORE_DIR=base_dir / "core",
            DATA_DIR=data_dir,
            MODEL_PATH=models_dir / "yolo" / "best.pt",
            SKU_DIR=sku_library_dir,
            SKU_FEATURES=sku_library_dir / "sku_features.npy",
            SKU_INDEX=sku_library_dir / "sku_library.csv",
            SKU_IMAGES_DIR=sku_library_dir / "images",
            SKU_MODEL_PATH=sku_model_path,
            ULTRALYTICS_DIR=data_dir / ".ultralytics",
            YOLO_CONFIG_DIR=data_dir / ".yolo"
        )

        os.environ["ULTRALYTICS_CONFIG_DIR"] = str(self.paths.ULTRALYTICS_DIR)
        os.environ["YOLO_CONFIG_DIR"] = str(self.paths.YOLO_CONFIG_DIR)

    def reload(self):
        """重新加载配置"""
        self._initialized = False
        self.__init__()


config = Config()
