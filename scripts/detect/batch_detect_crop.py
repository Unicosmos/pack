"""
批量YOLO检测并裁剪脚本
支持批量处理图片文件夹，输出裁剪结果和检测信息JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.detector import BoxDetector
from core.utils.logger import logger
from config import config


def process_single_image(
    detector: BoxDetector,
    image_path: str,
    output_dir: str,
    save_crops: bool = True,
    min_area: int = 0,
    img_size: int = None
) -> Dict[str, Any]:
    """
    处理单张图片

    Args:
        detector: YOLO检测器
        image_path: 图片路径
        output_dir: 输出目录
        save_crops: 是否保存裁剪图片
        min_area: 最小面积阈值（像素）
        img_size: 图片缩放尺寸（None表示不缩放）

    Returns:
        检测结果字典
    """
    image_name = Path(image_path).stem
    image_ext = Path(image_path).suffix

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"读取图片失败 {image_path}: {e}")
        return {
            "image_name": image_name + image_ext,
            "image_path": str(image_path),
            "status": "error",
            "error": str(e),
            "boxes": []
        }

    original_size = image.size

    if img_size:
        image = image.resize((img_size, img_size), Image.LANCZOS)

    result = detector.detect_single_image(image, return_cropped=save_crops, return_plot=False)

    detections = result.get("detections", [])

    output_folder = Path(output_dir) / image_name
    if save_crops and detections:
        output_folder.mkdir(parents=True, exist_ok=True)

    boxes_info = []

    for idx, det in enumerate(detections):
        bbox = det.get("bbox", [])
        conf = det.get("confidence", 0.0)
        cls_id = det.get("class_id", 0)
        cls_name = det.get("class", "box")

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if min_area > 0 and area < min_area:
            continue

        box_info = {
            "bbox": bbox,
            "confidence": round(conf, 4),
            "class_id": cls_id,
            "class_name": cls_name
        }

        if save_crops and det.get("cropped_image"):
            crop_filename = f"{image_name}_{idx}.jpg"
            crop_path = output_folder / crop_filename

            try:
                cropped_img = det["cropped_image"]
                cropped_img.save(str(crop_path), quality=95)
                box_info["crop_path"] = str(crop_path)
            except Exception as e:
                logger.error(f"保存裁剪图片失败 {crop_path}: {e}")
                box_info["crop_error"] = str(e)

        boxes_info.append(box_info)

    result_dict = {
        "image_name": image_name + image_ext,
        "image_path": str(image_path),
        "status": "success",
        "original_size": list(original_size),
        "total_boxes": len(detections),
        "boxes": boxes_info
    }

    if img_size and img_size != original_size[0]:
        result_dict["processed_size"] = list(image.size)

    return result_dict


def batch_process(
    input_dir: str,
    output_dir: str,
    model_path: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
    device: str = None,
    half: bool = False,
    max_det: int = 300,
    classes: List[int] = None,
    img_size: int = None,
    min_area: int = 0,
    save_crops: bool = True,
    extensions: List[str] = None
) -> Dict[str, Any]:
    """
    批量处理图片文件夹

    Args:
        input_dir: 输入图片文件夹
        output_dir: 输出目录
        model_path: YOLO模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        imgsz: 推理图像尺寸
        device: 运行设备
        half: 是否使用半精度
        max_det: 最大检测数量
        classes: 类别过滤列表
        img_size: 图片缩放尺寸
        min_area: 最小面积阈值
        save_crops: 是否保存裁剪图片
        extensions: 支持的图片扩展名列表

    Returns:
        批量处理结果
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载YOLO模型: {model_path}")
    detector = BoxDetector(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
        half=half,
        max_det=max_det,
        classes=classes
    )

    if not detector.is_ready():
        raise RuntimeError("检测器加载失败")

    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        logger.warning(f"未找到图片文件: {input_dir}")
        return {
            "status": "error",
            "message": "未找到图片文件",
            "total_images": 0,
            "results": []
        }

    logger.info(f"找到 {len(image_files)} 张图片")

    results = []
    success_count = 0
    error_count = 0
    total_boxes = 0

    for idx, img_file in enumerate(image_files, 1):
        logger.info(f"处理 [{idx}/{len(image_files)}]: {img_file.name}")

        try:
            result = process_single_image(
                detector=detector,
                image_path=str(img_file),
                output_dir=output_dir,
                save_crops=save_crops,
                min_area=min_area,
                img_size=img_size
            )

            results.append(result)

            if result["status"] == "success":
                success_count += 1
                total_boxes += result["total_boxes"]
            else:
                error_count += 1

        except Exception as e:
            logger.error(f"处理失败 {img_file}: {e}")
            results.append({
                "image_name": img_file.name,
                "image_path": str(img_file),
                "status": "error",
                "error": str(e),
                "boxes": []
            })
            error_count += 1

    summary = {
        "status": "completed",
        "config": {
            "model_path": model_path,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "imgsz": imgsz,
            "device": device,
            "half": half,
            "max_det": max_det,
            "classes": classes,
            "img_size": img_size,
            "min_area": min_area,
            "save_crops": save_crops
        },
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_images": len(image_files),
        "success_count": success_count,
        "error_count": error_count,
        "total_boxes": total_boxes,
        "results": results
    }

    json_path = output_path / "detection_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="批量YOLO检测并裁剪脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python batch_detect_crop.py -i ./images -o ./output -m ./best.pt
  python batch_detect_crop.py -i ./images -o ./output -m ./best.pt --conf 0.5 --iou 0.6
  python batch_detect_crop.py -i ./images -o ./output -m ./best.pt --imgsz 1280 --device 0 --half
  python batch_detect_crop.py -i ./images -o ./output -m ./best.pt --classes 0 2 --max-det 100
  python batch_detect_crop.py -i ./images -o ./output -m ./best.pt --min-area 1000 --no-save
        """
    )

    parser.add_argument("-i", "--input", required=True, help="输入图片文件夹路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录路径")
    parser.add_argument("-m", "--model", default=None, help="YOLO模型路径（默认使用配置中的模型）")

    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值（默认: 0.4）")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU阈值（默认: 0.5）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸（默认: 640）")
    parser.add_argument("--device", type=str, default=None, help="运行设备，如 cpu, 0, 0,1,2,3（默认自动）")
    parser.add_argument("--half", action="store_true", help="启用半精度推理（需GPU）")
    parser.add_argument("--max-det", type=int, default=300, help="最大检测数量（默认: 300）")
    parser.add_argument("--classes", nargs='+', type=int, default=None, 
                       help="只检测指定类别ID，如 --classes 0 2 表示只检测第1和第3类")
    parser.add_argument("--img-size", type=int, default=None, help="图片缩放尺寸（默认: 不缩放）")
    parser.add_argument("--min-area", type=int, default=0, help="最小面积阈值，像素（默认: 0）")

    parser.add_argument("--no-save", action="store_true", help="不保存裁剪图片")
    parser.add_argument("--ext", nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'],
                       help="支持的图片扩展名（默认: .jpg .jpeg .png .bmp）")

    args = parser.parse_args()

    model_path = args.model if args.model else str(config.paths.MODEL_PATH)

    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("批量YOLO检测并裁剪")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"置信度阈值: {args.conf}")
    logger.info(f"IOU阈值: {args.iou}")
    logger.info(f"推理图像尺寸: {args.imgsz}")
    logger.info(f"运行设备: {args.device if args.device else 'auto'}")
    logger.info(f"半精度: {'是' if args.half else '否'}")
    logger.info(f"最大检测数: {args.max_det}")
    logger.info(f"类别过滤: {args.classes if args.classes else '所有类别'}")
    logger.info(f"图片尺寸: {args.img_size if args.img_size else '原始尺寸'}")
    logger.info(f"最小面积: {args.min_area}")
    logger.info(f"保存裁剪: {'否' if args.no_save else '是'}")
    logger.info("=" * 60)

    try:
        summary = batch_process(
            input_dir=args.input,
            output_dir=args.output,
            model_path=model_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            max_det=args.max_det,
            classes=args.classes,
            img_size=args.img_size,
            min_area=args.min_area,
            save_crops=not args.no_save,
            extensions=args.ext
        )

        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"总图片数: {summary['total_images']}")
        logger.info(f"成功: {summary['success_count']}")
        logger.info(f"失败: {summary['error_count']}")
        logger.info(f"总检测框: {summary['total_boxes']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"批量处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
