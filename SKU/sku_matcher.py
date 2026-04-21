"""
SKU匹配验证模块
对新图片中的箱体进行自动识别与SKU匹配

改进：
1. 支持SKU多张图片特征存储（多角度/多面匹配）
2. 使用padding替代CenterCrop，避免裁掉关键区域

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


def resize_with_padding(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    保持比例缩放并padding到目标尺寸
    
    Args:
        image: PIL图像
        target_size: 目标尺寸
    
    Returns:
        处理后的图像
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    
    # 灰色padding（128是中性灰）
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    result = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    result.paste(resized, (pad_w, pad_h))
    return result


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
            sku_db_dir: SKU库目录（包含sku_database.json, sku_features.npy）
            device: 推理设备
        """
        self.device = self._get_device(device)
        
        print("加载模型...")
        self.yolo = YOLO(yolo_path)
        print(f"  YOLO模型: {yolo_path}")
        
        self.resnet, self.feature_dim = self._load_resnet()
        print(f"  ResNet50 特征维度: {self.feature_dim}")
        
        # 改进的预处理：padding替代CenterCrop
        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: resize_with_padding(x, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        sku_db_dir = Path(sku_db_dir)
        
        # 查找 SKU 数据库文件
        db_path = sku_db_dir / "sku_database.json"
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                self.sku_database = json.load(f)
            print(f"  SKU数据库: {db_path}")
        else:
            # 如果没有数据库文件，从文件夹结构推断
            print("  未找到 sku_database.json，从 SKU 文件夹结构构建数据库...")
            self.sku_database = self._build_database_from_folders(sku_db_dir)
        
        # PCA 模型是可选的（需要在加载特征之前加载）
        pca_path = sku_db_dir / "pca_model.joblib"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            print(f"  PCA模型: {pca_path}")
            print(f"  PCA降维维度: {self.pca.n_components_}")
        else:
            self.pca = None
            print("  未找到PCA模型，将使用原始特征")
        
        # 查找预计算的特征文件
        feat_path = sku_db_dir / "sku_features.npy"
        index_path = sku_db_dir / "sku_feature_index.json"
        
        if feat_path.exists() and index_path.exists():
            self.sku_features = np.load(feat_path)
            with open(index_path, 'r', encoding='utf-8') as f:
                self.sku_feature_index = json.load(f)
            
            # 检查特征维度是否匹配（考虑PCA）
            expected_dim = self.pca.n_components_ if self.pca else self.feature_dim
            
            # 检查是否需要重新提取：
            # 1. 维度不匹配
            # 2. 特征矩阵形状异常（如第一维等于特征维度，可能是维度反了）
            need_reextract = False
            if self.sku_features.shape[1] != expected_dim:
                print(f"  警告: 特征维度不匹配 (当前: {self.sku_features.shape[1]}, 期望: {expected_dim})")
                need_reextract = True
            elif self.sku_features.shape[0] == self.feature_dim or self.sku_features.shape[0] == 2048:
                # 第一维等于特征维度，很可能是维度反了
                print(f"  警告: 特征矩阵形状异常 {self.sku_features.shape}，可能维度反了")
                need_reextract = True
            
            if need_reextract:
                # 删除旧文件，强制重新提取
                print(f"  删除旧特征文件并重新提取...")
                if feat_path.exists():
                    feat_path.unlink()
                if index_path.exists():
                    index_path.unlink()
                self.sku_features, self.sku_feature_index = self._extract_features_from_folders(sku_db_dir)
            
            print(f"  SKU特征矩阵: {self.sku_features.shape}")
            print(f"  SKU数量: {len(self.sku_feature_index)}")
        else:
            # 如果没有预计算特征，从 SKU 图片中提取特征
            print("  未找到特征文件，从 SKU 图片提取特征...")
            self.sku_features, self.sku_feature_index = self._extract_features_from_folders(sku_db_dir)
            print(f"  已提取 {len(self.sku_features)} 个特征向量")
            print(f"  SKU数量: {len(self.sku_feature_index)}")
        
        print(f"  设备: {self.device}")
    
    def _build_database_from_folders(self, sku_db_dir: Path) -> Dict[str, Any]:
        """
        从文件夹结构构建 SKU 数据库
        
        Args:
            sku_db_dir: SKU 库目录
            
        Returns:
            SKU 数据库字典
        """
        sku_dirs = [d for d in sku_db_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        skus = []
        for sku_dir in sorted(sku_dirs):
            sku_id = sku_dir.name
            # 获取该 SKU 下的所有图片
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            images = []
            for ext in image_extensions:
                images.extend(sku_dir.glob(f'*{ext}'))
                images.extend(sku_dir.glob(f'*{ext.upper()}'))
            
            images = sorted(set(images))
            
            sku_info = {
                "sku_id": sku_id,
                "name": sku_id,
                "folder": str(sku_dir),
                "images": [img.name for img in images],
                "image_count": len(images)
            }
            skus.append(sku_info)
        
        return {"skus": skus, "version": "2.0", "generated": "from_folders"}
    
    def _extract_features_from_folders(self, sku_db_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        从 SKU 文件夹中的所有图片提取特征（支持多张图片）
        
        Args:
            sku_db_dir: SKU 库目录
            
        Returns:
            (特征矩阵, 特征索引字典)
            特征索引格式: {sku_id: {"start": int, "end": int, "count": int}}
        """
        sku_dirs = [d for d in sku_db_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        sku_dirs = sorted(sku_dirs, key=lambda x: x.name)
        
        all_features = []
        sku_feature_index = {}
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        for sku_dir in sku_dirs:
            sku_id = sku_dir.name
            
            # 收集该SKU下的所有图片
            images = []
            for ext in image_extensions:
                images.extend(sku_dir.glob(f'*{ext}'))
                images.extend(sku_dir.glob(f'*{ext.upper()}'))
            images = sorted(set(images))
            
            start_idx = len(all_features)
            sku_features = []
            
            for img_path in images:
                try:
                    sample_image = Image.open(img_path)
                    if sample_image.mode != 'RGB':
                        sample_image = sample_image.convert('RGB')
                    
                    input_tensor = self.preprocess(sample_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        feat = self.resnet(input_tensor).squeeze().cpu().numpy()
                    
                    # L2 归一化
                    feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
                    sku_features.append(feat_norm)
                except Exception as e:
                    print(f"    警告: 无法处理 {img_path.name}: {e}")
            
            if sku_features:
                all_features.extend(sku_features)
                sku_feature_index[sku_id] = {
                    "start": start_idx,
                    "end": start_idx + len(sku_features),
                    "count": len(sku_features)
                }
                print(f"    SKU {sku_id}: 提取 {len(sku_features)} 个特征向量")
            else:
                # 如果没有图片，使用零向量
                zero_feat = np.zeros(self.feature_dim)
                all_features.append(zero_feat)
                sku_feature_index[sku_id] = {
                    "start": start_idx,
                    "end": start_idx + 1,
                    "count": 1
                }
                print(f"    SKU {sku_id}: 无有效图片，使用零向量")
        
        features = np.array(all_features, dtype=np.float32)
        
        # 如果有PCA模型，对SKU特征也应用PCA
        if self.pca is not None and len(features) > 0:
            features = self.pca.transform(features)
            print(f"  PCA降维: {self.feature_dim} -> {features.shape[1]}")
        
        # L2归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        features = features / norms
        
        # 更新特征维度
        actual_feature_dim = features.shape[1]
        
        # 保存特征和索引
        np.save(sku_db_dir / "sku_features.npy", features)
        with open(sku_db_dir / "sku_feature_index.json", 'w', encoding='utf-8') as f:
            json.dump(sku_feature_index, f, indent=2, ensure_ascii=False)
        print(f"  特征已保存到: {sku_db_dir / 'sku_features.npy'}")
        print(f"  索引已保存到: {sku_db_dir / 'sku_feature_index.json'}")
        
        return features, sku_feature_index
    
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
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        YOLO检测箱体
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            (检测结果列表, 原始图片numpy数组)
        """
        results = self.yolo.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        boxes = []
        orig_img = None
        if len(results) > 0 and results[0].boxes is not None:
            pred = results[0]
            orig_img = pred.orig_img
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
        
        return boxes, orig_img
    
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
        匹配SKU（支持多张图片特征，取每个SKU的最大相似度）
        
        Args:
            features: 特征矩阵 (N, D)
            threshold: 匹配阈值
            
        Returns:
            匹配结果列表
        """
        if len(self.sku_features) == 0:
            return [{"sku_id": "Unknown", "similarity": 0.0, "status": "no_sku"} for _ in range(len(features))]
        
        # 计算所有相似度
        similarities = np.dot(features, self.sku_features.T)  # [N_boxes, N_all_features]
        
        results = []
        for i in range(len(features)):
            # 对每个SKU，取其所有图片特征的最大相似度
            max_sim_per_sku = {}
            for sku_id, idx_info in self.sku_feature_index.items():
                start, end = idx_info["start"], idx_info["end"]
                if start < len(similarities[i]):
                    max_sim = similarities[i, start:min(end, len(similarities[i]))].max()
                    max_sim_per_sku[sku_id] = max_sim
            
            if not max_sim_per_sku:
                results.append({
                    "sku_id": "Unknown",
                    "similarity": 0.0,
                    "status": "no_match"
                })
                continue
            
            # 找最佳匹配
            best_sku = max(max_sim_per_sku, key=max_sim_per_sku.get)
            best_sim = max_sim_per_sku[best_sku]
            
            # 总是返回最佳匹配的 SKU，即使相似度低于阈值
            if best_sim >= threshold:
                results.append({
                    "sku_id": best_sku,
                    "similarity": float(best_sim),
                    "status": "matched"
                })
            else:
                results.append({
                    "sku_id": best_sku,  # 显示最相似的 SKU
                    "similarity": float(best_sim),
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
            # 使用更大的字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            font = ImageFont.load_default()
            font_small = font
            font_large = font
        
        matched_count = sum(1 for r in match_results if r["status"] == "matched")
        unmatched_count = len(match_results) - matched_count
        
        stats_text = f"Total: {len(boxes)}  Matched: {matched_count}  Unknown: {unmatched_count}"
        draw.rectangle([5, 5, 350, 35], fill=(0, 0, 0, 200))
        draw.text((10, 10), stats_text, fill=(255, 255, 255), font=font_small)
        
        for box, result in zip(boxes, match_results):
            x1, y1, x2, y2 = box["bbox"]
            conf = box.get("confidence", 0.0)
            
            if result["status"] == "matched":
                color = (0, 255, 0)
                bg_color = (0, 200, 0)
            else:
                color = (255, 0, 0)
                bg_color = (200, 0, 0)
            
            # 显示 SKU 编号、置信度和相似度
            label = f"SKU: {result['sku_id']}\nConf: {conf:.2f}\nSim: {result['similarity']:.2f}"
            
            # 增加框体线条粗细
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            # 计算文本框位置，确保在图片范围内
            text_lines = label.split('\n')
            max_line_width = max(draw.textlength(line, font=font) for line in text_lines)
            text_height = len(text_lines) * 30  # 每行约30像素高度
            
            # 文本框位置：在框体上方，如果空间不足则放在下方
            text_x = x1
            text_y = y1 - text_height - 20
            if text_y < 60:  # 确保不与顶部统计信息重叠
                text_y = y2 + 20
            
            # 绘制文本背景，使用不透明的彩色背景
            draw.rectangle([text_x - 10, text_y - 10, text_x + max_line_width + 10, text_y + text_height + 10], 
                          fill=bg_color)
            
            # 绘制文本，使用黑色文字
            for i, line in enumerate(text_lines):
                draw.text((text_x, text_y + i * 30), line, fill=(0, 0, 0), font=font)
        
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
        
        print(f"  [1] 检测箱体...")
        boxes, orig_img = self.detect_boxes(image_path, conf_threshold)
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
        
        # 将 numpy 数组转换为 PIL Image 用于特征提取
        # 确保是 RGB 格式（YOLO 返回的可能是 BGR）
        if orig_img.shape[2] == 3:
            orig_img = orig_img[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(orig_img)
        
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
    # 处理单张图片
    python sku_matcher.py -i test.jpg --sku-db ./sku_output --output ./results/test_result.jpg
    
    # 处理单张图片（指定输出）
    python sku_matcher.py --image test.jpg --sku-db ./sku_output --output ./results/test_result.jpg
    
    # 调整匹配阈值
    python sku_matcher.py -i test.jpg --sku-db ./sku_output --threshold 0.8
    
    # 批量处理文件夹中的图片
    python sku_matcher.py -i ./test_images --sku-db ./sku_output -o ./results
    
    # 递归处理子文件夹（包括所有嵌套子文件夹）
    python sku_matcher.py -i ./test_images --sku-db ./sku_output -o ./results --recursive
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='单张图片路径或图片文件夹路径（批量处理）'
    )
    
    parser.add_argument(
        '--recursive', '-R','-r',
        action='store_true',
        help='当输入为文件夹时，递归扫描子文件夹中的图片'
    )
    
    parser.add_argument(
        '--yolo',
        type=str,
        default=None,
        help='YOLO模型路径（默认使用SKU目录下的best.pt）'
    )
    
    parser.add_argument(
        '--sku-db','-s',
        type=str,
        default='./sku_output',
        help='SKU库目录 (默认: ./sku_output)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（单张图片模式）或输出目录（批量模式，默认: ./results）'
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
        # 首先尝试使用默认的best.pt路径
        default_yolo_path = "./best.pt"
        if Path(default_yolo_path).exists():
            yolo_path = default_yolo_path
        else:
            # 然后尝试在SKU库目录中查找
            yolo_path = str(Path(args.sku_db) / "best.pt")
    
    print("=" * 70)
    print("SKU匹配验证模块 v2.0")
    print("改进: 多图特征存储 + Padding预处理")
    print("=" * 70)
    
    matcher = SKUMatcher(
        yolo_path=yolo_path,
        sku_db_dir=args.sku_db,
        device=args.device
    )
    
    if args.image:
        input_path = Path(args.image)
        
        # 判断是文件还是目录
        if input_path.is_file():
            # 单张图片处理
            print("\n检测到单张图片，开始处理...")
            output_path = args.output or "./result.jpg"
            result = matcher.process_image(
                str(input_path),
                output_path,
                conf_threshold=args.conf,
                match_threshold=args.threshold
            )
            
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {json_path}")
        
        elif input_path.is_dir():
            # 文件夹批量处理
            print("\n检测到文件夹，开始批量处理...")
            output_dir = Path(args.output or "./results")
            output_dir.mkdir(parents=True, exist_ok=True)
            

            
            # 支持的图片格式
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
            # 扫描图片
            if args.recursive:
                image_paths = []
                for ext in extensions:
                    image_paths.extend(input_path.rglob(f'*{ext}'))
                    image_paths.extend(input_path.rglob(f'*{ext.upper()}'))
            else:
                image_paths = []
                for ext in extensions:
                    image_paths.extend(input_path.glob(f'*{ext}'))
                    image_paths.extend(input_path.glob(f'*{ext.upper()}'))
            
            image_paths = sorted(set(image_paths))
            print(f"\n找到 {len(image_paths)} 张图片")
            
            if len(image_paths) == 0:
                print("未找到任何图片文件")
                return
            
            all_results = []
            matched_count = 0
            unmatched_count = 0
            
            for idx, img_path in enumerate(image_paths):
                print(f"\n[{idx+1}/{len(image_paths)}] {img_path.name}")
                output_path = output_dir / f"{img_path.stem}_result.jpg"
                result = matcher.process_image(
                    str(img_path),
                    str(output_path),
                    conf_threshold=args.conf,
                    match_threshold=args.threshold
                )
                
                # 统计匹配结果
                for detail in result.get("details", []):
                    if detail["status"] == "matched":
                        matched_count += 1
                    else:
                        unmatched_count += 1
                
                all_results.append(result)
            
            summary = {
                "total_images": len(image_paths),
                "total_boxes": sum(r["total_boxes"] for r in all_results),
                "total_matched": matched_count,
                "total_unmatched": unmatched_count,
                "results": all_results
            }
            
            summary_path = output_dir / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print("\n" + "=" * 70)
            print("批量处理完成")
            print("=" * 70)
            print(f"总图片数: {len(image_paths)}")
            print(f"总箱体数: {summary['total_boxes']}")
            print(f"匹配成功: {matched_count}")
            print(f"未匹配: {unmatched_count}")
            print(f"匹配率: {matched_count / max(summary['total_boxes'], 1) * 100:.1f}%")
            print(f"结果目录: {output_dir}")
            print("=" * 70)
        
        else:
            print(f"错误: 输入路径不存在: {input_path}")
    
    else:
        print("错误: 请指定输入图片路径 (--image 或 -i)")
        print("使用 --help 查看帮助信息")


if __name__ == '__main__':
    main()
