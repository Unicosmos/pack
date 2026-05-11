"""
Pack Web API - 重构版
基于FastAPI的图片检测和SKU匹配服务
集成新模块：config, models, core, utils
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SKU"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))

try:
    from box_detector import BoxDetector as OriginalBoxDetector
    HAS_ORIGINAL_MODULES = True
except ImportError as e:
    HAS_ORIGINAL_MODULES = False
    print(f"警告: 原始SKU模块导入失败: {e}")

from config import config
from models.schemas import (
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
from utils.image_utils import (
    filter_small_boxes, 
    crop_box, 
    resize_with_padding, 
    image_to_base64,
    process_uploaded_image,
    generate_crops_base64,
)
from utils.logger import logger


sys.path.insert(0, str(Path(__file__).parent / "core"))
from detector import BoxDetector
from matcher import SKUMatcher


detector: Optional[BoxDetector] = None
matcher: Optional[SKUMatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, matcher

    logger.info("=" * 50)
    logger.info("Pack Web API 启动中...")

    cfg = config

    if cfg.paths.MODEL_PATH.exists():
        logger.info(f"加载检测模型: {cfg.paths.MODEL_PATH}")
        try:
            detector = BoxDetector(str(cfg.paths.MODEL_PATH), conf_threshold=cfg.model.CONF_THRESHOLD)
            logger.info("  BoxDetector加载成功")
        except Exception as e:
            logger.error(f"  BoxDetector加载失败: {e}")
    else:
        logger.warning(f"  警告: 模型文件不存在: {cfg.paths.MODEL_PATH}")

    if cfg.paths.SKU_DIR.exists():
        logger.info(f"加载SKU库: {cfg.paths.SKU_DIR}")
        try:
            matcher = SKUMatcher(
                str(cfg.paths.MODEL_PATH),
                str(cfg.paths.SKU_DIR),
                match_threshold=cfg.match.MATCH_THRESHOLD,
                ratio_threshold=cfg.match.RATIO_THRESHOLD,
                sku_model_path=str(cfg.paths.SKU_MODEL_PATH) if cfg.paths.SKU_MODEL_PATH else None
            )
            if matcher.is_ready():
                logger.info("  SKUMatcher加载成功")
            else:
                logger.warning("  SKUMatcher未就绪（可能缺少特征文件）")
        except Exception as e:
            logger.error(f"  SKUMatcher加载失败: {e}")
            matcher = None
    else:
        logger.info("  SKU库目录不存在，匹配功能将不可用")

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

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

sku_images_dir = Path(__file__).parent.parent.parent / "sku_library" / "images"
if sku_images_dir.exists():
    app.mount("/static/sku_images", StaticFiles(directory=str(sku_images_dir)), name="sku_images")


def get_sku_count() -> int:
    """获取SKU数量"""
    if matcher and matcher.is_ready():
        sku_ids = set()
        for item in matcher.sku_info:
            sku_ids.add(item.get('sku_id', ''))
        return len(sku_ids)
    return 0


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数验证错误处理"""
    logger.warning(f"参数验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "detail": "请求参数验证失败",
            "errors": exc.errors()
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
    detector_ready = detector is not None and detector.is_ready()
    matcher_ready = matcher is not None and matcher.is_ready()
    sku_count_val = get_sku_count()

    if detector is None:
        status = "init"
        message = "系统初始化中，检测模型未加载"
    elif not detector_ready:
        status = "error"
        message = "检测模型加载失败"
    elif matcher is None or not matcher_ready:
        status = "partial"
        message = "检测就绪，但SKU匹配功能不可用"
    else:
        status = "ok"
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
    if detector is None or not detector.is_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载，请检查模型文件是否存在")

    try:
        contents = await file.read()
        image = process_uploaded_image(contents)

        result = detector.detect_single_image(image, return_cropped=True, return_plot=True)

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

        # 优先使用YOLO自带的可视化
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
    if matcher is None or not matcher.is_ready():
        raise HTTPException(status_code=503, detail="SKU匹配器未加载，请检查SKU库是否存在")

    try:
        contents = await file.read()
        image = process_uploaded_image(contents)

        resized = resize_with_padding(image, target_size=config.model.INPUT_SIZE)
        features = matcher.extract_feature(resized)

        result = matcher.match_sku(features, threshold=match_threshold, ratio_threshold=ratio_threshold)

        top5_labels = [TopLabel(label=t['label'], similarity=t['similarity'], image_name=t.get('image_name', ''), sku_id=t.get('sku_id', ''), sku_name=t.get('sku_name', '')) for t in result.top5_labels] if result.top5_labels else []

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
    if detector is None or not detector.is_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载，请检查模型文件是否存在")

    try:
        contents = await file.read()
        image = process_uploaded_image(contents)

        result = detector.detect_single_image(image, return_cropped=True, return_plot=True)

        boxes = result.get("detections", [])
        plot_image = result.get("plot_image", None)
        
        if not boxes:
            return DetectAndMatchResponse(
                success=True,
                count=0,
                matched_count=0,
                low_conf_count=0,
                unmatched_count=0,
                boxes=[],
                matches=[],
                image_with_boxes=None,
                sku_matcher_enabled=matcher is not None and matcher.is_ready()
            )

        boxes = filter_small_boxes(
            boxes,
            image.size,
            min_area_ratio=config.model.MIN_AREA_RATIO,
            min_pixel_area=config.model.MIN_PIXEL_AREA
        )

        match_results = []
        sku_matcher_enabled = matcher is not None and matcher.is_ready()

        if sku_matcher_enabled and boxes:
            try:
                features = []
                for box in boxes:
                    cropped = crop_box(image, box.get("bbox", []))
                    if cropped:
                        resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                        feat = matcher.extract_feature(resized)
                        features.append(feat)
                    else:
                        features.append(None)

                for feat in features:
                    if feat is None:
                        match_results.append(MatchResult(
                            sku_id=None,
                            similarity=0.0,
                            ratio=None,
                            status="unmatched",
                            top5_labels=[]
                        ))
                    else:
                        mr = matcher.match_sku(feat, threshold=match_threshold)
                        match_results.append(mr)
            except Exception as e:
                print(f"匹配失败: {e}")
                sku_matcher_enabled = False

        if not sku_matcher_enabled:
            match_results = [None] * len(boxes)

        # 优先使用YOLO自带的可视化图片，如果没有则使用自定义绘制
        if plot_image:
            result_image = plot_image
        else:
            result_image, _ = draw_detection_result(image, boxes, match_results)
        
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

        match_infos = []
        matched_count = 0
        low_conf_count = 0
        unmatched_count = 0

        for mr in match_results:
            if mr is None:
                match_infos.append(None)
                unmatched_count += 1
            else:
                top5 = [TopLabel(label=t['label'], similarity=t['similarity'], image_name=t.get('image_name', ''), sku_id=t.get('sku_id', ''), sku_name=t.get('sku_name', '')) for t in mr.top5_labels] if mr.top5_labels else []
                match_infos.append(MatchInfo(
                    sku_id=mr.sku_id,
                    similarity=mr.similarity,
                    ratio=mr.ratio,
                    status=mr.status,
                    top5_labels=top5
                ))
                if mr.status == "matched":
                    matched_count += 1
                elif mr.status == "low_conf":
                    low_conf_count += 1
                else:
                    unmatched_count += 1

        return DetectAndMatchResponse(
            success=True,
            count=len(boxes),
            matched_count=matched_count,
            low_conf_count=low_conf_count,
            unmatched_count=unmatched_count,
            boxes=box_infos,
            crops=crops_base64,
            image_with_boxes=img_base64,
            matches=match_infos,
            sku_matcher_enabled=sku_matcher_enabled
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/api/skus", response_model=SKUListResponse)
async def get_sku_list():
    """获取SKU列表"""
    if matcher is None or not matcher.is_ready():
        return SKUListResponse(success=True, skus=[], count=0)

    sku_map = {}
    for item in matcher.sku_info:
        sku_id = item.get('sku_id', '')
        if sku_id:
            if sku_id not in sku_map:
                sku_map[sku_id] = {
                    'sku_id': sku_id,
                    'sku_name': item.get('sku_name', sku_id),
                    'labels': []
                }
            sku_map[sku_id]['labels'].append(item.get('label', ''))

    skus = [
        SKUInfo(
            sku_id=sku_id,
            sku_name=info['sku_name'],
            label_count=len(info['labels']),
            image_count=len(info['labels'])
        )
        for sku_id, info in sku_map.items()
    ]

    return SKUListResponse(success=True, skus=skus, count=len(skus))


@app.get("/api/sku-image/{sku_id}/{image_name}")
async def get_sku_image(sku_id: str, image_name: str):
    """获取SKU图片"""
    sku_images_dir = Path(__file__).parent.parent.parent / "sku_library" / "images"
    image_path = sku_images_dir / sku_id / image_name
    
    if not image_path.exists():
        return JSONResponse(status_code=404, content={"detail": "图片不存在"})
    
    try:
        with open(image_path, "rb") as f:
            content = f.read()
        
        if image_name.lower().endswith(".jpg") or image_name.lower().endswith(".jpeg"):
            media_type = "image/jpeg"
        elif image_name.lower().endswith(".png"):
            media_type = "image/png"
        else:
            media_type = "application/octet-stream"
        
        return Response(content=content, media_type=media_type)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)