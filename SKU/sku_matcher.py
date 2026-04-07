"""
SKU匹配验证模块
对新图片中的箱体进行自动识别与SKU匹配

使用方法:
    python sku_matcher.py --image test.jpg --sku-db ./sku_output --output ./results
    
    # 调整匹配阈值
    python sku_matcher.py --image test.jpg --sku-db ./sku_output --threshold 0.8
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib

from ultralytics import YOLO


class SKUMatcher:
    """SKU匹配器"""
    
    def __init__(
        self,
        yolo_path: str,
        sku_db_dir: str,
        device: str = None
    ):
        """
        初始化匹配器
        
        Args:
            yolo_path: YOLO模型路径
            sku_db_dir: SKU库目录（包含sku_database.json, sku_features.npy, pca_model.joblib）
            device: 推理设备
        """
        self.device = self._get_device(device)
        
        print("加载模型...")
        self.yolo = YOLO(yolo_path)
        print(f"  YOLO模型: {yolo_path}")
        
        self.resnet, self.feature_dim = self._load_resnet()
        print(f"  ResNet50 特征维度: {self.feature_dim}")
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        sku_db_dir = Path(sku_db_dir)
        
        db_path = sku_db_dir / "sku_database.json"
        with open(db_path, 'r', encoding='utf-8') as f:
            self.sku_database = json.load(f)
        print(f"  SKU数据库: {db_path}")
        
        feat_path = sku_db_dir / "sku_features.npy"
        self.sku_features = np.load(feat_path)
        print(f"  SKU特征矩阵: {self.sku_features.shape}")
        
        pca_path = sku_db_dir / "pca_model.joblib"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            print(f"  PCA模型: {pca_path}")
        else:
            self.pca = None
            print("  警告: 未找到PCA模型，将使用原始特征")
        
        self.sku_id_map = {i: sku["sku_id"] for i, sku in enumerate(self.sku_database["skus"])}
        
        print(f"  已加载 {len(self.sku_database['skus'])} 个SKU")
        print(f"  设备: {self.device}")
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """确定推理设备"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    
    def _load_resnet(self) -> Tuple[nn.Module, int]:
        """加载ResNet50模型（去掉分类头）"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        backbone = nn.Sequential(*list(model.children())[:-1])
        backbone.to(self.device)
        backbone.eval()
        return backbone, feature_dim
    
    def detect_boxes(
        self,
        image_path: str,
        conf_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        YOLO检测箱体
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        results = self.yolo.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            pred = results[0]
            for i in range(len(pred.boxes)):
                box = pred.boxes.xyxy[i].cpu().numpy()
                conf = float(pred.boxes.conf[i].cpu().numpy())
                cls_id = int(pred.boxes.cls[i].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, f"class_{cls_id}")
                
                boxes.append({
                    "bbox": list(map(int, box)),
                    "confidence": conf,
                    "class": cls_name
                })
        
        return boxes
    
    def extract_features(
        self,
        image: Image.Image,
        boxes: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        提取箱体特征
        
        Args:
            image: PIL图像
            boxes: 检测框列表
            
        Returns:
            特征矩阵 (N, feature_dim)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        features = []
        
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(self.feature_dim))
                continue
            
            cropped = image.crop((x1, y1, x2, y2))
            
            input_tensor = self.preprocess(cropped).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.resnet(input_tensor).squeeze().cpu().numpy()
            
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        features = features / norms
        
        if self.pca is not None and features.shape[0] > 0:
            features = self.pca.transform(features)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            features = features / norms
        
        return features
    
    def match_sku(
        self,
        features: np.ndarray,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        匹配SKU
        
        Args:
            features: 特征矩阵 (N, D)
            threshold: 匹配阈值
            
        Returns:
            匹配结果列表
        """
        if len(self.sku_features) == 0:
            return [{"sku_id": "Unknown", "similarity": 0.0, "status": "no_sku"} for _ in range(len(features))]
        
        similarities = np.dot(features, self.sku_features.T)
        
        results = []
        for i in range(len(features)):
            sim_scores = similarities[i]
            max_idx = np.argmax(sim_scores)
            max_sim = float(sim_scores[max_idx])
            
            if max_sim >= threshold:
                sku_id = self.sku_id_map.get(max_idx, "Unknown")
                results.append({
                    "sku_id": sku_id,
                    "similarity": max_sim,
                    "status": "matched"
                })
            else:
                results.append({
                    "sku_id": "Unknown",
                    "similarity": max_sim,
                    "status": "unmatched"
                })
        
        return results
    
    def visualize_results(
        self,
        image: Image.Image,
        boxes: List[Dict[str, Any]],
        match_results: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        可视化结果
        
        Args:
            image: PIL图像
            boxes: 检测框列表
            match_results: 匹配结果列表
            output_path: 输出路径
        """
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        matched_count = sum(1 for r in match_results if r["status"] == "matched")
        unmatched_count = len(match_results) - matched_count
        
        stats_text = f"Total: {len(boxes)}  Matched: {matched_count}  Unknown: {unmatched_count}"
        draw.rectangle([5, 5, 350, 30], fill=(0, 0, 0, 180))
        draw.text((10, 8), stats_text, fill=(255, 255, 255), font=font_small)
        
        for box, result in zip(boxes, match_results):
            x1, y1, x2, y2 = box["bbox"]
            
            if result["status"] == "matched":
                color = (0, 255, 0)
                label = f"{result['sku_id']} ({result['similarity']:.2f})"
            else:
                color = (255, 0, 0)
                label = f"Unknown ({result['similarity']:.2f})"
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font_small)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font_small)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)
        print(f"可视化结果已保存: {output_path}")
    
    def process_image(
        self,
        image_path: str,
        output_path: str,
        conf_threshold: float = 0.5,
        match_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        完整处理流程
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            conf_threshold: 检测置信度阈值
            match_threshold: SKU匹配阈值
            
        Returns:
            处理结果字典
        """
        print(f"\n处理图片: {image_path}")
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"  [1] 检测箱体...")
        boxes = self.detect_boxes(image_path, conf_threshold)
        print(f"      检测到 {len(boxes)} 个箱体")
        
        if len(boxes) == 0:
            result = {
                "image_path": image_path,
                "total_boxes": 0,
                "matched_boxes": 0,
                "unmatched_boxes": 0,
                "details": []
            }
            print("  未检测到箱体")
            return result
        
        print(f"  [2] 提取特征...")
        features = self.extract_features(image, boxes)
        
        print(f"  [3] 匹配SKU...")
        match_results = self.match_sku(features, match_threshold)
        
        matched_count = sum(1 for r in match_results if r["status"] == "matched")
        unmatched_count = len(match_results) - matched_count
        
        print(f"      匹配成功: {matched_count}, 未匹配: {unmatched_count}")
        
        details = []
        for box, result in zip(boxes, match_results):
            details.append({
                "bbox": box["bbox"],
                "sku_id": result["sku_id"],
                "similarity": round(result["similarity"], 4),
                "status": result["status"]
            })
            print(f"      - {result['sku_id']}: {result['similarity']:.4f}")
        
        print(f"  [4] 生成可视化...")
        self.visualize_results(image, boxes, match_results, output_path)
        
        result = {
            "image_path": image_path,
            "total_boxes": len(boxes),
            "matched_boxes": matched_count,
            "unmatched_boxes": unmatched_count,
            "details": details
        }
        
        return result


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='SKU匹配验证模块',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python sku_matcher.py --image test.jpg --sku-db ./sku_output --output ./results/test_result.jpg
    
    # 调整匹配阈值
    python sku_matcher.py --image test.jpg --sku-db ./sku_output --threshold 0.8
    
    # 批量处理
    python sku_matcher.py --images ./test_images --sku-db ./sku_output --output-dir ./results
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='单张图片路径'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        help='图片目录路径（批量处理）'
    )
    
    parser.add_argument(
        '--yolo',
        type=str,
        default=None,
        help='YOLO模型路径（默认使用SKU目录下的best.pt）'
    )
    
    parser.add_argument(
        '--sku-db',
        type=str,
        default='./sku_output',
        help='SKU库目录 (默认: ./sku_output)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（单张图片模式）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='输出目录（批量模式，默认: ./results）'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='检测置信度阈值 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,
        help='SKU匹配阈值 (默认: 0.85)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='推理设备 (如 cuda:0, cpu)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    yolo_path = args.yolo
    if yolo_path is None:
        yolo_path = str(Path(args.sku_db) / "best.pt")
        if not Path(yolo_path).exists():
            yolo_path = "./best.pt"
    
    print("=" * 70)
    print("SKU匹配验证模块")
    print("=" * 70)
    
    matcher = SKUMatcher(
        yolo_path=yolo_path,
        sku_db_dir=args.sku_db,
        device=args.device
    )
    
    if args.image:
        output_path = args.output or "./result.jpg"
        result = matcher.process_image(
            args.image,
            output_path,
            conf_threshold=args.conf,
            match_threshold=args.threshold
        )
        
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {json_path}")
    
    elif args.images:
        images_dir = Path(args.images)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(images_dir.glob(f'*{ext}'))
            image_paths.extend(images_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        print(f"\n找到 {len(image_paths)} 张图片")
        
        all_results = []
        for idx, img_path in enumerate(image_paths):
            print(f"\n[{idx+1}/{len(image_paths)}]")
            output_path = output_dir / f"{img_path.stem}_result.jpg"
            result = matcher.process_image(
                str(img_path),
                str(output_path),
                conf_threshold=args.conf,
                match_threshold=args.threshold
            )
            all_results.append(result)
        
        summary = {
            "total_images": len(image_paths),
            "total_boxes": sum(r["total_boxes"] for r in all_results),
            "total_matched": sum(r["matched_boxes"] for r in all_results),
            "total_unmatched": sum(r["unmatched_boxes"] for r in all_results),
            "results": all_results
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n汇总结果已保存: {summary_path}")
    
    else:
        print("错误: 请指定 --image 或 --images 参数")
        sys.exit(1)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
