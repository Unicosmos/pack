"""
结果可视化模块
在原图上绘制检测框和匹配结果
"""

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from core.utils.image_utils import crop_box, resize_with_padding, image_to_base64


COLOR_MATCHED = (0, 200, 0)
COLOR_LOW_CONF = (255, 140, 0)
COLOR_UNMATCHED = (200, 0, 0)
COLOR_NO_MATCH = (150, 150, 150)


def get_box_color(match_result: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    if match_result is None:
        return COLOR_NO_MATCH

    status = match_result.get("status", "")
    if status == "matched":
        return COLOR_MATCHED
    elif status == "low_conf":
        return COLOR_LOW_CONF
    else:
        return COLOR_UNMATCHED


def get_box_label(match_result: Optional[Dict[str, Any]], confidence: float, box_idx: int = 0) -> str:
    label_parts = [f"#{box_idx + 1}"]
    
    if match_result is None:
        label_parts.append(f"Conf: {confidence:.2f}")
    else:
        status = match_result.get("status", "")
        sku_id = match_result.get("sku_id", "Unknown")
        similarity = match_result.get("similarity", 0.0)

        if status == "matched":
            label_parts.append(f"{sku_id} ({similarity:.2f})")
        elif status == "low_conf":
            label_parts.append(f"{sku_id}? ({similarity:.2f})")
        else:
            label_parts.append("Unknown")
    
    return " | ".join(label_parts)


def draw_single_box(
    draw: ImageDraw.ImageDraw,
    bbox: List[int],
    color: Tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    line_width: int = 4
) -> None:
    x1, y1, x2, y2 = bbox

    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    text_bbox = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, max(0, y1 - text_height)), label, fill=(255, 255, 255), font=font)


def draw_detection_result(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    match_results: List[Optional[Dict[str, Any]]] = None
) -> Tuple[Image.Image, List[str]]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    image_width, image_height = image.size

    crops_base64 = []

    for i, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        confidence = box.get("confidence", 0.0)

        if not bbox or len(bbox) != 4:
            crops_base64.append(None)
            continue

        match_result = None
        if match_results and i < len(match_results):
            match_result = match_results[i]

        color = get_box_color(match_result)
        label = get_box_label(match_result, confidence, i)

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        base_font_size = max(28, min(56, int(min(box_width, box_height) / 3.5)))
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size)
            except Exception:
                font = ImageFont.load_default()

        line_width = max(10, min(20, int(min(box_width, box_height) / 20)))

        draw_single_box(draw, bbox, color, label, font, line_width)

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
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    image_width, image_height = image.size

    for i, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        confidence = box.get("confidence", 0.0)

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        base_font_size = max(20, min(40, int(min(box_width, box_height) / 5)))
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size)
            except Exception:
                font = ImageFont.load_default()

        line_width = max(7, min(15, int(min(box_width, box_height) / 25)))

        color = COLOR_NO_MATCH
        label = f"#{i + 1} | Conf: {confidence:.2f}"

        draw_single_box(draw, bbox, color, label, font, line_width)

    return result_image


def draw_detection_result_from_db(
    image: Image.Image,
    detection_boxes: List[Any],
    match_results: Dict[int, Any] = None
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    if match_results is None:
        match_results = {}

    for i, db_box in enumerate(detection_boxes):
        x1, y1, x2, y2 = db_box.bbox_x1, db_box.bbox_y1, db_box.bbox_x2, db_box.bbox_y2
        bbox = [x1, y1, x2, y2]
        confidence = db_box.confidence

        mr = match_results.get(db_box.id)

        color = get_box_color(mr)
        
        # Convert match result to dict format for label function
        match_result_dict = None
        if mr:
            match_result_dict = {
                "sku_id": mr.sku_id or "Unknown",
                "similarity": mr.similarity or 0.0,
                "status": mr.status or "unmatched"
            }
        
        label = get_box_label(match_result_dict, confidence, i)

        box_width = x2 - x1
        box_height = y2 - y1

        base_font_size = max(28, min(56, int(min(box_width, box_height) / 3.5)))
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size)
            except Exception:
                font = ImageFont.load_default()

        line_width = max(10, min(20, int(min(box_width, box_height) / 20)))

        draw_single_box(draw, bbox, color, label, font, line_width)

    return result_image