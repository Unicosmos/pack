import math
import io
import base64
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np


def calculate_box_area(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_box_area_ratio(bbox: List[int], image_size: Tuple[int, int]) -> float:
    box_area = calculate_box_area(bbox)
    image_area = image_size[0] * image_size[1]
    return box_area / image_area if image_area > 0 else 0


def calculate_aspect_ratio(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return float('inf')
    return width / height


def filter_small_boxes(
    boxes: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    min_area_ratio: float = 0.01,
    min_pixel_area: int = 2500,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 5.0
) -> List[Dict[str, Any]]:
    filtered = []
    img_width, img_height = image_size

    for box in boxes:
        bbox = box.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            continue

        area_ratio = calculate_box_area_ratio(bbox, image_size)
        pixel_area = calculate_box_area(bbox)
        aspect_ratio = calculate_aspect_ratio(bbox)

        if area_ratio < min_area_ratio:
            continue
        if pixel_area < min_pixel_area:
            continue
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        filtered.append(box)

    return filtered


def resize_with_padding(
    image: Image.Image,
    target_size: int = 224,
    fill_color: Tuple[int, int, int] = (128, 128, 128)
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_width, orig_height = image.size
    scale = min(target_size / orig_width, target_size / orig_height)

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    result = Image.new("RGB", (target_size, target_size), fill_color)

    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    result.paste(resized, (paste_x, paste_y))

    return result


def crop_box(
    image: Image.Image,
    bbox: List[int],
    expand_ratio: float = 0.0
) -> Optional[Image.Image]:
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

    orig_width, orig_height = image.size

    if expand_ratio > 0:
        box_width = x2 - x1
        box_height = y2 - y1
        expand_x = int(box_width * expand_ratio)
        expand_y = int(box_height * expand_ratio)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(orig_width, x2 + expand_x)
        y2 = min(orig_height, y2 + expand_y)

    try:
        cropped = image.crop((x1, y1, x2, y2))
        return cropped
    except Exception:
        return None


def image_to_base64(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 85
) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception:
        return None


def process_uploaded_image(contents: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def generate_crops_base64(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    target_size: int = 224
) -> List[Optional[str]]:
    crops_base64 = []
    for box in boxes:
        cropped = crop_box(image, box.get("bbox", []))
        if cropped:
            resized = resize_with_padding(cropped, target_size=target_size)
            crops_base64.append(image_to_base64(resized))
        else:
            crops_base64.append(None)
    return crops_base64


def build_box_info_list(boxes: List[Dict[str, Any]]) -> List[Any]:
    return [
        {
            "bbox": b.get("bbox", []),
            "confidence": b.get("confidence", 0.0),
            "class_id": b.get("class_id", 0),
            "class_name": b.get("class_name", "box")
        }
        for b in boxes
    ]
