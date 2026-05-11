"""
结果可视化模块
在原图上绘制检测框和匹配结果
"""

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import crop_box, resize_with_padding, image_to_base64


COLOR_MATCHED = (0, 255, 0)
COLOR_LOW_CONF = (255, 165, 0)
COLOR_UNMATCHED = (255, 0, 0)
COLOR_NO_MATCH = (128, 128, 128)


def get_box_color(match_result: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    根据匹配结果获取检测框颜色

    Args:
        match_result: 匹配结果字典，包含status字段

    Returns:
        RGB颜色元组
    """
    if match_result is None:
        return COLOR_NO_MATCH

    status = match_result.get("status", "")
    if status == "matched":
        return COLOR_MATCHED
    elif status == "low_conf":
        return COLOR_LOW_CONF
    else:
        return COLOR_UNMATCHED


def get_box_label(match_result: Optional[Dict[str, Any]], confidence: float) -> str:
    """
    根据匹配结果获取检测框标签

    Args:
        match_result: 匹配结果字典
        confidence: 检测置信度

    Returns:
        标签字符串
    """
    if match_result is None:
        return f"Conf: {confidence:.2f}"

    status = match_result.get("status", "")
    sku_id = match_result.get("sku_id", "Unknown")
    similarity = match_result.get("similarity", 0.0)

    if status == "matched":
        return f"SKU: {sku_id} ({similarity:.2f})"
    elif status == "low_conf":
        return f"SKU: {sku_id}? ({similarity:.2f})"
    else:
        return "Unknown"


def draw_single_box(
    draw: ImageDraw.ImageDraw,
    bbox: List[int],
    color: Tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont
) -> None:
    """在图像上绘制单个检测框和标签"""
    x1, y1, x2, y2 = bbox

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    text_bbox = draw.textbbox((x1, max(0, y1 - 25)), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, max(0, y1 - 25)), label, fill=(255, 255, 255), font=font)


def draw_detection_result(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    match_results: List[Optional[Dict[str, Any]]] = None
) -> Tuple[Image.Image, List[str]]:
    """
    在原图上绘制检测框和匹配结果

    Args:
        image: PIL Image对象
        boxes: 检测框列表，每个元素包含bbox、confidence等
        match_results: 匹配结果列表，与boxes一一对应

    Returns:
        Tuple[Image.Image, List[str]]:
        - 绘制后的图像
        - 裁剪图Base64列表
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

    crops_base64 = []

    for i, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        confidence = box.get("confidence", 0.0)

        match_result = None
        if match_results and i < len(match_results):
            match_result = match_results[i]

        color = get_box_color(match_result)
        label = get_box_label(match_result, confidence)

        draw_single_box(draw, bbox, color, label, font)

        cropped = crop_box(image, bbox)
        if cropped:
            resized = resize_with_padding(cropped, target_size=224)
            crops_base64.append(image_to_base64(resized))
        else:
            crops_base64.append(None)

    return result_image, crops_base64


def draw_boxes_only(
    image: Image.Image,
    boxes: List[Dict[str, Any]]
) -> Image.Image:
    """
    仅绘制检测框（不显示匹配结果）

    Args:
        image: PIL Image对象
        boxes: 检测框列表

    Returns:
        绘制后的图像
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

    for box in boxes:
        bbox = box.get("bbox", [])
        confidence = box.get("confidence", 0.0)

        if not bbox or len(bbox) != 4:
            continue

        color = COLOR_NO_MATCH
        label = f"Conf: {confidence:.2f}"

        draw_single_box(draw, bbox, color, label, font)

    return result_image
