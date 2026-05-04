"""
图像处理工具模块
包含小框过滤、resize、裁剪等功能
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np


def calculate_box_area(bbox: List[int]) -> int:
    """计算检测框面积"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_box_area_ratio(bbox: List[int], image_size: Tuple[int, int]) -> float:
    """计算检测框占图像面积的比例"""
    box_area = calculate_box_area(bbox)
    image_area = image_size[0] * image_size[1]
    return box_area / image_area if image_area > 0 else 0


def calculate_aspect_ratio(bbox: List[int]) -> float:
    """计算检测框宽高比"""
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
    """
    过滤小框和宽高比异常的框

    过滤条件:
    - 面积占比 < min_area_ratio (默认1%)
    - 像素面积 < min_pixel_area (默认2500)
    - 宽高比 < min_aspect_ratio 或 > max_aspect_ratio

    Args:
        boxes: 检测框列表，每个元素包含bbox、confidence等
        image_size: 图像尺寸 (width, height)
        min_area_ratio: 最小面积占比
        min_pixel_area: 最小像素面积
        min_aspect_ratio: 最小宽高比
        max_aspect_ratio: 最大宽高比

    Returns:
        过滤后的框列表
    """
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
    """
    将图像resize到目标尺寸，保持宽高比，不足部分用灰色填充

    Args:
        image: PIL Image对象
        target_size: 目标尺寸，默认224
        fill_color: 填充颜色，默认灰色(128,128,128)

    Returns:
        resize后的图像
    """
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
    """
    根据检测框坐标裁剪图像

    Args:
        image: PIL Image对象
        bbox: 检测框坐标 [x1, y1, x2, y2]
        expand_ratio: 裁剪区域扩展比例（相对于框大小）

    Returns:
        裁剪后的图像，如果坐标无效则返回None
    """
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
    """
    将PIL图像转换为Base64编码字符串

    Args:
        image: PIL Image对象
        format: 图像格式，默认JPEG
        quality: 质量，默认85

    Returns:
        Base64编码字符串
    """
    import io
    import base64

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    """
    将Base64编码字符串转换为PIL图像

    Args:
        base64_str: Base64编码字符串

    Returns:
        PIL Image对象，转换失败返回None
    """
    import io
    import base64

    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception:
        return None


def normalize_image_for_vit(image: Image.Image) -> np.ndarray:
    """
    对图像进行ImageNet标准归一化（用于ViT输入）

    Args:
        image: PIL Image对象

    Returns:
        归一化后的numpy数组，形状为(3, H, W)
    """
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image)
    return tensor.numpy()
