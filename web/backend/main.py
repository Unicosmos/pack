"""
Pack Web API - 重构版
基于FastAPI的图片检测和SKU匹配服务
集成核心模块：core, config, models, utils
"""

import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import uuid
from datetime import datetime

from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils.pytorch_utils import init_pytorch_env
init_pytorch_env()

from config import config
from schemas.schemas import (
    HealthResponse,
    DetectResponse,
    DetectAndMatchResponse,
    MatchResponse,
    SKUListResponse,
    BoxInfo,
    MatchInfo,
    TopLabel,
    SKUInfo,
    ErrorResponse,
)
from core.visualizer import draw_detection_result, draw_boxes_only
from core.utils.image_utils import (
    filter_small_boxes,
    crop_box,
    resize_with_padding,
    image_to_base64,
    process_uploaded_image,
    generate_crops_base64,
)
from core.utils.logger import logger

# 引入服务层
from services.detect_match_service import DetectMatchService
from repositories.task_repository import TaskRepository

from database import init_db, SessionLocal, get_db
from models.task import Task
from api.task import router as task_router
from api.sku import router as sku_router
from api.sku_review import router as sku_review_router
from api.logs import router as logs_router
from api.build import router as build_router


# 服务实例
detect_match_service = DetectMatchService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Pack Web API 启动中...")

    logger.info("初始化数据库...")
    init_db()

    logger.info("初始化检测匹配服务...")
    detect_match_service.initialize()

    logger.info("=" * 50)

    yield

    logger.info("Pack Web API 关闭")


app = FastAPI(
    title="Pack Web API",
    description="地堆箱货检测和SKU匹配服务",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task_router)
app.include_router(sku_router)
app.include_router(sku_review_router)
app.include_router(logs_router)
app.include_router(build_router)

# 先挂载子路径，再挂载父路径
if config.paths.SKU_IMAGES_DIR.exists():
    app.mount("/static/sku_images", StaticFiles(directory=str(config.paths.SKU_IMAGES_DIR)), name="sku_images")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 挂载 SKU 审核用到的文件夹 - 挂载整个训练 SKU 目录（使用不冲突的路径）
sku_root = Path(__file__).parent.parent.parent / "training" / "sku"
if sku_root.exists():
    app.mount("/sku-static", StaticFiles(directory=str(sku_root)), name="sku_root")


def get_sku_count() -> int:
    """获取SKU数量"""
    return detect_match_service.get_sku_count()


def serialize_value(value):
    """递归序列化值，处理不可序列化的对象"""
    if isinstance(value, (ValueError, Exception)):
        return str(value)
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    else:
        return value

def serialize_errors(errors):
    """序列化错误信息，处理不可序列化的对象"""
    return [serialize_value(error) for error in errors]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数验证错误处理"""
    logger.warning(f"参数验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "detail": "请求参数验证失败",
            "errors": serialize_errors(exc.errors())
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    logger.warning(f"HTTP异常 {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未捕获异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": f"服务器内部错误: {str(exc)}",
            "status_code": 500
        }
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    detector_ready = detect_match_service.is_detection_ready()
    matcher_ready = detect_match_service.is_match_ready()
    sku_count_val = get_sku_count()

    logger.info(f"Health check - detector_ready: {detector_ready}, matcher_ready: {matcher_ready}")

    if not detector_ready:
        status = "init"
        message = "系统初始化中，检测模型未加载"
    elif not matcher_ready:
        status = "ready"
        message = "检测就绪，SKU匹配功能未配置"
    else:
        status = "ready"
        message = "系统正常运行"

    return HealthResponse(
        status=status,
        message=message,
        detector_ready=detector_ready,
        matcher_ready=matcher_ready,
        sku_count=sku_count_val,
        model_path=str(config.paths.MODEL_PATH),
        sku_dir=str(config.paths.SKU_DIR)
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """仅检测接口（不进行SKU匹配）"""
    logger.info(f"detect_image called - detector_ready: {detect_match_service.is_detection_ready()}")
    if not detect_match_service.is_detection_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载，请检查模型文件是否存在")

    try:
        contents = await file.read()
        image = process_uploaded_image(contents)
        result = detect_match_service.detection_service.detector.detect_single_image(
            image, return_cropped=True, return_plot=True
        )

        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)

        if not boxes:
            return DetectResponse(
                success=True,
                count=0,
                boxes=[],
                crops=[],
                image_with_boxes=None
            )

        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )

        if plot_image:
            result_image = plot_image
        else:
            result_image = draw_boxes_only(image, boxes)
        img_base64 = image_to_base64(result_image)

        crops_base64 = generate_crops_base64(image, boxes, target_size=config.model.INPUT_SIZE)

        box_infos = [
            BoxInfo(
                bbox=b.get("bbox", []),
                confidence=b.get("confidence", 0.0),
                class_id=b.get("class_id", 0),
                class_name=b.get("class_name", "box")
            )
            for b in boxes
        ]

        return DetectResponse(
            success=True,
            count=len(boxes),
            boxes=box_infos,
            crops=crops_base64,
            image_with_boxes=img_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/match", response_model=MatchResponse)
async def match_image(
    file: UploadFile = File(...),
    match_threshold: float = 0.85,
    ratio_threshold: float = 1.2
):
    """仅SKU匹配接口"""
    if not detect_match_service.is_match_ready():
        raise HTTPException(status_code=503, detail="SKU匹配器未加载，请检查SKU库是否存在")

    try:
        contents = await file.read()
        image = process_uploaded_image(contents)

        resized = resize_with_padding(image, target_size=config.model.INPUT_SIZE)
        features = detect_match_service.match_service.matcher.extract_feature(resized)
        result = detect_match_service.match_service.matcher.match_sku(
            features, threshold=match_threshold, ratio_threshold=ratio_threshold
        )

        top5_labels = [
            TopLabel(
                label=t['label'], 
                similarity=t['similarity'], 
                image_path=t.get('image_path', ''), 
                sku_id=t.get('sku_id', ''), 
                sku_name=t.get('sku_name', '')
            ) 
            for t in result.top5_labels
        ] if result.top5_labels else []

        return MatchResponse(
            success=True,
            sku_id=result.sku_id,
            similarity=result.similarity,
            ratio=result.ratio,
            status=result.status,
            top5_labels=top5_labels
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"匹配失败: {str(e)}")


@app.post("/api/detect-and-match", response_model=DetectAndMatchResponse)
async def detect_and_match_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    match_threshold: float = 0.85
):
    """检测+匹配接口（主接口）"""
    logger.info(f"detect_and_match_image called - detector_ready: {detect_match_service.is_detection_ready()}")
    if not detect_match_service.is_detection_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载，请检查模型文件是否存在")

    try:
        contents = await file.read()
        result = detect_match_service.detect_and_match(
            contents, 
            file.filename,
            conf_threshold=conf_threshold,
            match_threshold=match_threshold
        )

        return DetectAndMatchResponse(
            success=True,
            count=result["count"],
            matched_count=result["matched_count"],
            low_conf_count=result["low_conf_count"],
            unmatched_count=result["unmatched_count"],
            boxes=result["boxes"],
            crops=result["crops"],
            image_with_boxes=result["image_with_boxes"],
            task_id=result["task_id"],
            matches=result["matches"],
            sku_matcher_enabled=result["sku_matcher_enabled"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/api/skus", response_model=SKUListResponse)
async def get_sku_list():
    """获取SKU列表"""
    if not detect_match_service.is_match_ready():
        return SKUListResponse(success=True, skus=[], count=0)

    sku_list = detect_match_service.get_sku_list()
    skus = [
        SKUInfo(
            sku_id=item['sku_id'],
            sku_name=item['sku_name'],
            label_count=item['label_count'],
            image_count=item['image_count']
        )
        for item in sku_list
    ]

    return SKUListResponse(success=True, skus=skus, count=len(skus))


@app.get("/api/sku-image")
async def get_sku_image(path: str):
    """获取SKU图片或任务上传的图片
    Args:
        path: CSV中的完整路径，如 'images\\000001\\1 (112)_001.jpg' 或任务上传路径
    """
    from urllib.parse import unquote
    path = unquote(path)
    
    # 先尝试作为任务上传的图片路径
    image_path = Path(path)
    
    # 如果不是绝对路径，尝试在SKU库中查找
    if not image_path.is_absolute():
        image_path = config.paths.SKU_DIR / path
    
    if not image_path.exists():
        return JSONResponse(status_code=404, content={"detail": f"图片不存在: {image_path}"})

    try:
        with open(image_path, "rb") as f:
            content = f.read()

        ext = image_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            media_type = "image/jpeg"
        elif ext == ".png":
            media_type = "image/png"
        elif ext == ".bmp":
            media_type = "image/bmp"
        elif ext == ".gif":
            media_type = "image/gif"
        elif ext == ".webp":
            media_type = "image/webp"
        else:
            media_type = "application/octet-stream"

        return Response(content=content, media_type=media_type)
    except Exception as e:
        logger.error(f"读取图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取图片失败: {str(e)}")


@app.get("/")
async def root():
    """首页"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Pack Web API",
        "docs": "/docs",
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
