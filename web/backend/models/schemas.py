"""
Pydantic 数据模型
定义API请求/响应的数据结构
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class BoxInfo(BaseModel):
    bbox: List[int] = Field(..., description="检测框坐标 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    class_id: int = Field(0, description="类别ID")
    class_name: str = Field("box", description="类别名称")


class TopLabel(BaseModel):
    label: str = Field(..., description="标签名称")
    similarity: float = Field(..., description="相似度")


class MatchInfo(BaseModel):
    sku_id: Optional[str] = Field(None, description="匹配的SKU编号")
    similarity: Optional[float] = Field(None, description="Top-1相似度")
    ratio: Optional[float] = Field(None, description="相似度比值")
    status: str = Field(..., description="匹配状态: matched/low_conf/unmatched")
    top5_labels: Optional[List[TopLabel]] = Field(None, description="Top-5候选标签")


class HealthResponse(BaseModel):
    status: str = Field(..., description="系统状态: ok/error/init")
    detector_ready: bool = Field(False, description="检测器是否就绪")
    matcher_ready: bool = Field(False, description="匹配器是否就绪")
    sku_count: int = Field(0, description="SKU库数量")
    model_path: str = Field("", description="模型路径")
    sku_dir: str = Field("", description="SKU库路径")


class DetectResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    count: int = Field(0, description="检测数量")
    boxes: List[BoxInfo] = Field(default_factory=list, description="检测框列表")
    crops: List[str] = Field(default_factory=list, description="裁剪图Base64列表")
    image_with_boxes: Optional[str] = Field(None, description="带框图像Base64")


class DetectAndMatchResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    count: int = Field(0, description="检测数量")
    matched_count: int = Field(0, description="已匹配数量")
    low_conf_count: int = Field(0, description="低置信数量")
    unmatched_count: int = Field(0, description="未匹配数量")
    boxes: List[BoxInfo] = Field(default_factory=list, description="检测框列表")
    crops: List[str] = Field(default_factory=list, description="裁剪图Base64列表")
    image_with_boxes: Optional[str] = Field(None, description="带框图像Base64")
    matches: List[Optional[MatchInfo]] = Field(default_factory=list, description="匹配结果列表")
    sku_matcher_enabled: bool = Field(True, description="SKU匹配功能是否启用")


class MatchResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    matches: List[MatchInfo] = Field(default_factory=list, description="匹配结果列表")


class SKUInfo(BaseModel):
    sku_id: str = Field(..., description="SKU编号")
    sku_name: str = Field("", description="SKU名称")
    label_count: int = Field(0, description="标签数量")
    image_count: int = Field(0, description="图片数量")


class SKUListResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    skus: List[SKUInfo] = Field(default_factory=list, description="SKU列表")
    count: int = Field(0, description="SKU数量")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误信息")
