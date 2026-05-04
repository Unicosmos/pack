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

import io
import base64

from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SKU"))

try:
    from box_detector import BoxDetector as OriginalBoxDetector
    HAS_ORIGINAL_MODULES = True
except ImportError:
    HAS_ORIGINAL_MODULES = False
    print("警告: 原始SKU模块导入失败")

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
from utils.image_utils import filter_small_boxes, crop_box, resize_with_padding, image_to_base64


sys.path.insert(0, str(Path(__file__).parent / "core"))
from matcher import SKUMatcher, BoxDetector


detector: Optional[BoxDetector] = None
matcher: Optional[SKUMatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, matcher

    print("=" * 50)
    print("Pack Web API 启动中...")

    cfg = config

    if cfg.paths.MODEL_PATH.exists():
        print(f"加载检测模型: {cfg.paths.MODEL_PATH}")
        try:
            detector = BoxDetector(str(cfg.paths.MODEL_PATH), conf_threshold=cfg.model.CONF_THRESHOLD)
            print("  BoxDetector加载成功")
        except Exception as e:
            print(f"  BoxDetector加载失败: {e}")
    else:
        print(f"  警告: 模型文件不存在: {cfg.paths.MODEL_PATH}")

    if cfg.paths.SKU_DIR.exists():
        print(f"加载SKU库: {cfg.paths.SKU_DIR}")
        try:
            matcher = SKUMatcher(
                str(cfg.paths.MODEL_PATH),
                str(cfg.paths.SKU_DIR),
                match_threshold=cfg.match.MATCH_THRESHOLD,
                ratio_threshold=cfg.match.RATIO_THRESHOLD
            )
            if matcher.is_ready():
                print("  SKUMatcher加载成功")
            else:
                print("  SKUMatcher未就绪（可能缺少特征文件）")
        except Exception as e:
            print(f"  SKUMatcher加载失败: {e}")
            matcher = None
    else:
        print("  SKU库目录不存在，匹配功能将不可用")

    print("=" * 50)

    yield

    print("Pack Web API 关闭")


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


def get_sku_count() -> int:
    """获取SKU数量"""
    if matcher and matcher.is_ready():
        sku_ids = set()
        for item in matcher.sku_index:
            sku_ids.add(item.get('sku_id', ''))
        return len(sku_ids)
    return 0


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    detector_ready = detector is not None and detector.is_ready()
    matcher_ready = matcher is not None and matcher.is_ready()

    if detector is None:
        status = "init"
    elif not detector_ready:
        status = "error"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        detector_ready=detector_ready,
        matcher_ready=matcher_ready,
        sku_count=get_sku_count(),
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
        raise HTTPException(status_code=503, detail="检测模型未加载")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        result = detector.detect_single_image(image, return_cropped=True)

        boxes = result.get("detections", [])
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

        result_image = draw_boxes_only(image, boxes)
        img_base64 = image_to_base64(result_image)

        crops_base64 = []
        for box in boxes:
            cropped = crop_box(image, box.get("bbox", []))
            if cropped:
                resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                crops_base64.append(image_to_base64(resized))
            else:
                crops_base64.append(None)

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


@app.post("/api/detect-and-match", response_model=DetectAndMatchResponse)
async def detect_and_match_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """检测+匹配接口（主接口）"""
    if detector is None or not detector.is_ready():
        raise HTTPException(status_code=503, detail="检测模型未加载")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        result = detector.detect_single_image(image, return_cropped=True)

        boxes = result.get("detections", [])
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
                        feat = matcher.extract_features(resized)
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
                        mr = matcher.match_sku(feat)
                        match_results.append(mr)
            except Exception as e:
                print(f"匹配失败: {e}")
                sku_matcher_enabled = False

        if not sku_matcher_enabled:
            match_results = [None] * len(boxes)

        result_image, crops_base64 = draw_detection_result(image, boxes, match_results)
        img_base64 = image_to_base64(result_image)

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
                top5 = [TopLabel(label=t['label'], similarity=t['similarity']) for t in mr.top5_labels] if mr.top5_labels else []
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
    for item in matcher.sku_index:
        sku_id = item.get('sku_id', '')
        if sku_id and sku_id not in sku_map:
            sku_map[sku_id] = {
                'sku_id': sku_id,
                'sku_name': item.get('sku_name', sku_id),
                'labels': []
            }
        if sku_id:
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
