"""
Pydantic 数据模型
定义API请求/响应的数据结构
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoxInfo(BaseModel):
    bbox: List[int] = Field(..., description="检测框坐标 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    class_id: int = Field(0, description="类别ID")
    class_name: str = Field("box", description="类别名称")


class DetectedBox(BaseModel):
    box_id: str = Field(..., description="检测框ID")
    bbox: List[int] = Field(..., description="检测框坐标 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    class_id: int = Field(0, description="类别ID")
    class_name: str = Field("box", description="类别名称")
    status: str = Field("approved", description="状态: approved/rejected")
    crop_base64: Optional[str] = Field(None, description="裁剪图Base64")


class DetectionResult(BaseModel):
    boxes: List[DetectedBox] = Field(default_factory=list, description="检测框列表")


class MatchResult(BaseModel):
    box_id: str = Field(..., description="检测框ID")
    sku_id: Optional[str] = Field(None, description="匹配的SKU编号")
    similarity: Optional[float] = Field(None, description="相似度")
    status: str = Field(..., description="匹配状态: matched/low_conf/unmatched")


class TopLabel(BaseModel):
    label: str = Field(..., description="标签名称")
    similarity: float = Field(..., description="相似度")
    image_path: Optional[str] = Field("", description="匹配图片路径")
    sku_id: Optional[str] = Field("", description="SKU编号")
    sku_name: Optional[str] = Field("", description="SKU名称")


class MatchInfo(BaseModel):
    sku_id: Optional[str] = Field(None, description="匹配的SKU编号")
    similarity: Optional[float] = Field(None, description="Top-1相似度")
    ratio: Optional[float] = Field(None, description="相似度比值")
    status: str = Field(..., description="匹配状态: matched/low_conf/unmatched")
    top5_labels: Optional[List[TopLabel]] = Field(None, description="Top-5候选标签")


class HealthResponse(BaseModel):
    status: str = Field(..., description="系统状态: ok/error/init/partial")
    message: str = Field("", description="状态描述信息")
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
    task_id: Optional[int] = Field(None, description="任务ID")


class MatchResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    sku_id: Optional[str] = Field(None, description="匹配的SKU编号")
    similarity: Optional[float] = Field(None, description="Top-1相似度")
    ratio: Optional[float] = Field(None, description="相似度比值")
    status: str = Field("", description="匹配状态: matched/low_conf/unmatched")
    top5_labels: List[TopLabel] = Field(default_factory=list, description="Top-5候选标签")


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


class TaskResponse(BaseModel):
    id: int
    image_name: str
    status: str
    detection_status: str
    review_status: str
    box_count: int
    matched_count: int
    unmatched_count: int
    result: Optional[Dict[str, Any]]
    created_at: str
    completed_at: Optional[str]

    class Config:
        from_attributes = True


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    box_count: Optional[int] = None
    matched_count: Optional[int] = None
    unmatched_count: Optional[int] = None
    detection_status: Optional[str] = None
    review_status: Optional[str] = None
    error_message: Optional[str] = None


class ReviewUpdate(BaseModel):
    boxes: List[Dict[str, Any]] = Field(..., description="审核后的检测框列表")


class ReviewResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    task_id: int
    approved_count: int = Field(0, description="审核通过数量")
    rejected_count: int = Field(0, description="审核拒绝数量")
    message: str = Field("", description="提示信息")


class DetectionBoxSchema(BaseModel):
    box_id: str = Field(..., description="检测框ID")
    bbox: List[int] = Field(..., description="检测框坐标 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    class_id: int = Field(0, description="类别ID")
    class_name: str = Field("box", description="类别名称")
    path: Optional[str] = Field(None, description="切图路径")
    status: str = Field("pending", description="状态: pending/approved/rejected")
    is_audited: bool = Field(False, description="是否已审核")
    created_at: Optional[str] = Field(None, description="创建时间")

    class Config:
        from_attributes = True


class OperationLogSchema(BaseModel):
    id: int
    entity_type: str
    entity_id: int
    action: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    operator_ip: Optional[str] = None
    operated_at: Optional[str] = None
    remark: Optional[str] = None

    class Config:
        from_attributes = True


class OperationLogListResponse(BaseModel):
    success: bool = Field(True, description="是否成功")
    logs: List[OperationLogSchema] = Field(default_factory=list, description="日志列表")
    total: int = Field(0, description="总数")
