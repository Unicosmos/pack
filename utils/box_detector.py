"""
箱体检测模块
使用 YOLOv8 模型检测图片中的箱体，输出边界框、类别及裁剪图像

使用方法:
    python box_detector.py --images /path/to/images --model /path/to/model.pt --conf 0.5
    
    # 单张图片
    python box_detector.py --images image.jpg --model runs/detect/yolov8n_cbam_xxx/weights/best.pt
    
    # 图片目录
    python box_detector.py --images /path/to/images --model runs/detect/yolov8n_cbam_xxx/weights/best.pt
    
    # 多张图片
    python box_detector.py --images img1.jpg img2.jpg img3.jpg --model best.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from PIL import Image
import numpy as np

from ultralytics import YOLO


class BoxDetector:
    """箱体检测器"""
    
    def __init__(
        self, 
        model_path: str, 
        conf_threshold: float = 0.5,
        device: str = None
    ):
        """
        初始化检测器
        
        Args:
            model_path: YOLOv8 模型路径
            conf_threshold: 置信度阈值
            device: 推理设备 (如 '0', 'cpu', '0,1')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        print(f"已加载模型: {model_path}")
        print(f"类别: {self.class_names}")
        print(f"置信度阈值: {conf_threshold}")
    
    def detect_single_image(
        self, 
        image_path: str,
        return_cropped: bool = True
    ) -> Dict[str, Any]:
        """
        对单张图片进行检测
        
        Args:
            image_path: 图片路径
            return_cropped: 是否返回裁剪图像
            
        Returns:
            检测结果字典
        """
        result = {
            "image_path": str(image_path),
            "image_width": None,
            "image_height": None,
            "detections": []
        }
        
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 - {image_path}")
            return result
        
        try:
            pil_image = Image.open(image_path)
            result["image_width"] = pil_image.width
            result["image_height"] = pil_image.height
        except Exception as e:
            print(f"警告: 无法读取图片 - {image_path}, 错误: {e}")
            return result
        
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0:
            return result
        
        pred = results[0]
        
        if pred.boxes is None or len(pred.boxes) == 0:
            return result
        
        boxes = pred.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            
            x1, y1, x2, y2 = map(int, box)
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4)
            }
            
            if return_cropped:
                x1_clamped = max(0, x1)
                y1_clamped = max(0, y1)
                x2_clamped = min(pil_image.width, x2)
                y2_clamped = min(pil_image.height, y2)
                
                if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                    cropped = pil_image.crop((x1_clamped, y1_clamped, x2_clamped, y2_clamped))
                    detection["cropped_image"] = cropped
                    detection["cropped_width"] = cropped.width
                    detection["cropped_height"] = cropped.height
                else:
                    detection["cropped_image"] = None
                    detection["cropped_width"] = 0
                    detection["cropped_height"] = 0
            
            result["detections"].append(detection)
        
        return result
    
    def detect_batch(
        self, 
        image_paths: List[str],
        return_cropped: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量检测多张图片
        
        Args:
            image_paths: 图片路径列表
            return_cropped: 是否返回裁剪图像
            verbose: 是否显示进度
            
        Returns:
            检测结果列表
        """
        results = []
        total = len(image_paths)
        
        for idx, img_path in enumerate(image_paths):
            if verbose:
                print(f"处理 [{idx+1}/{total}]: {img_path}")
            
            result = self.detect_single_image(img_path, return_cropped)
            results.append(result)
            
            if verbose:
                n_detections = len(result["detections"])
                print(f"  检测到 {n_detections} 个箱体")
        
        return results
    
    def detect_from_directory(
        self, 
        directory: str,
        extensions: List[str] = None,
        return_cropped: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检测目录下的所有图片
        
        Args:
            directory: 图片目录
            extensions: 图片扩展名列表
            return_cropped: 是否返回裁剪图像
            verbose: 是否显示进度
            
        Returns:
            检测结果列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        directory = Path(directory)
        if not directory.exists():
            print(f"错误: 目录不存在 - {directory}")
            return []
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        
        if verbose:
            print(f"找到 {len(image_paths)} 张图片")
        
        return self.detect_batch([str(p) for p in image_paths], return_cropped, verbose)


def save_cropped_images(
    results: List[Dict[str, Any]], 
    output_dir: str,
    verbose: bool = True
) -> None:
    """
    保存裁剪后的箱体图像
    
    Args:
        results: 检测结果列表
        output_dir: 输出目录
        verbose: 是否显示进度
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    for result in results:
        img_path = Path(result["image_path"])
        img_stem = img_path.stem
        
        for idx, det in enumerate(result["detections"]):
            if "cropped_image" in det and det["cropped_image"] is not None:
                cropped = det["cropped_image"]
                save_name = f"{img_stem}_box{idx}_{det['class']}_conf{det['confidence']:.2f}.jpg"
                save_path = output_dir / save_name
                cropped.save(save_path, quality=95)
                saved_count += 1
                
                if verbose:
                    print(f"保存: {save_path}")
    
    print(f"共保存 {saved_count} 张裁剪图像到 {output_dir}")


def print_detection_summary(results: List[Dict[str, Any]]) -> None:
    """打印检测结果摘要"""
    total_images = len(results)
    images_with_detections = sum(1 for r in results if len(r["detections"]) > 0)
    total_detections = sum(len(r["detections"]) for r in results)
    
    print("\n" + "=" * 60)
    print("检测结果摘要")
    print("=" * 60)
    print(f"总图片数: {total_images}")
    print(f"有检测结果的图片数: {images_with_detections}")
    print(f"总检测框数: {total_detections}")
    print(f"平均每张图片检测框数: {total_detections/total_images:.2f}" if total_images > 0 else "N/A")
    
    if total_detections > 0:
        confidences = [det["confidence"] for r in results for det in r["detections"]]
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度范围: [{min(confidences):.4f}, {max(confidences):.4f}]")
    
    print("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='箱体检测模块 - 使用 YOLOv8 检测图片中的箱体',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 单张图片
    python box_detector.py --images image.jpg --model best.pt
    
    # 图片目录
    python box_detector.py --images /path/to/images --model best.pt
    
    # 多张图片
    python box_detector.py --images img1.jpg img2.jpg --model best.pt
    
    # 指定置信度阈值和输出目录
    python box_detector.py --images /path/to/images --model best.pt --conf 0.6 --output ./cropped
        """
    )
    
    parser.add_argument(
        '--images', '-i',
        type=str,
        default="/root/source/data2/hyg/projects/hs/SKU/source",
        help='图片路径（单张、多张或目录）'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default="/root/source/data2/hyg/projects/hs/SKU/best.pt",
        help='YOLOv8 模型路径'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.5,
        help='置信度阈值 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default="0",
        help='推理设备 (如 0, cpu, 0,1)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="cropped_boxes",
        help='裁剪图像输出目录（可选）'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，减少输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    verbose = not args.quiet
    
    detector = BoxDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    image_paths = []
    images_arg = args.images
    
    if isinstance(images_arg, str):
        images_arg = [images_arg]
    
    for path_str in images_arg:
        path = Path(path_str)
        if path.is_dir():
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            for ext in extensions:
                image_paths.extend(path.glob(f'*{ext}'))
                image_paths.extend(path.glob(f'*{ext.upper()}'))
        elif path.exists():
            image_paths.append(path)
        else:
            print(f"警告: 路径不存在 - {path}")
    
    image_paths = sorted(set(image_paths))
    
    if len(image_paths) == 0:
        print("错误: 未找到任何图片")
        sys.exit(1)
    
    if verbose:
        print(f"\n共找到 {len(image_paths)} 张图片")
    
    results = detector.detect_batch(
        [str(p) for p in image_paths],
        return_cropped=True,
        verbose=verbose
    )
    
    if verbose:
        print_detection_summary(results)
    
    if args.output:
        save_cropped_images(results, args.output, verbose)
    
    return results


if __name__ == '__main__':
    main()
