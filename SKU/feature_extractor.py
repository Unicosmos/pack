"""
特征提取模块
使用预训练 ResNet50 模型提取箱体视觉特征

使用方法:
    python feature_extractor.py --input detection_results.pkl --output features_results.pkl
    
    # 与 box_detector.py 配合使用
    from box_detector import BoxDetector
    from feature_extractor import FeatureExtractor
    
    detector = BoxDetector("model.pt")
    results = detector.detect_batch(image_paths)
    
    extractor = FeatureExtractor()
    results = extractor.extract_features(results)
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


class FeatureExtractor:
    """基于 ResNet50 的特征提取器"""
    
    def __init__(
        self, 
        model_name: str = 'resnet50',
        device: str = None,
        pretrained: bool = True
    ):
        """
        初始化特征提取器
        
        Args:
            model_name: 模型名称 (resnet50, resnet101, resnet152)
            device: 推理设备 (如 'cuda:0', 'cpu')
            pretrained: 是否使用预训练权重
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = self._get_preprocess()
        
        print(f"已加载模型: {model_name}")
        print(f"特征维度: {self.feature_dim}")
        print(f"设备: {self.device}")
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """确定推理设备"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    
    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """加载预训练模型并移除分类头"""
        model_funcs = {
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            'resnet34': models.resnet34,
            'resnet18': models.resnet18,
        }
        
        if model_name not in model_funcs:
            raise ValueError(f"不支持的模型: {model_name}，可选: {list(model_funcs.keys())}")
        
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        if model_name == 'resnet50':
            model = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet152':
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.feature_dim = model.fc.in_features
        
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        
        return self.backbone
    
    def _get_preprocess(self) -> transforms.Compose:
        """获取图像预处理管道"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_single_image(self, image: Image.Image) -> np.ndarray:
        """
        从单张图像提取特征
        
        Args:
            image: PIL.Image 对象
            
        Returns:
            特征向量 (numpy array, shape: [feature_dim])
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def extract_from_detections(
        self, 
        detection_results: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从检测结果中提取特征
        
        Args:
            detection_results: box_detector 返回的检测结果列表
            verbose: 是否显示进度
            
        Returns:
            更新后的检测结果列表（包含 feature_vector 字段）
        """
        total_images = len(detection_results)
        total_boxes = sum(len(r.get("detections", [])) for r in detection_results)
        
        if verbose:
            print(f"开始提取特征...")
            print(f"总图片数: {total_images}")
            print(f"总箱体数: {total_boxes}")
        
        processed_boxes = 0
        
        for img_idx, result in enumerate(detection_results):
            detections = result.get("detections", [])
            
            for det_idx, detection in enumerate(detections):
                cropped_image = detection.get("cropped_image")
                
                if cropped_image is None:
                    detection["feature_vector"] = None
                    continue
                
                try:
                    feature_vector = self.extract_single_image(cropped_image)
                    detection["feature_vector"] = feature_vector
                except Exception as e:
                    print(f"警告: 提取特征失败 - 图片 {img_idx}, 箱体 {det_idx}: {e}")
                    detection["feature_vector"] = None
                
                processed_boxes += 1
                if verbose and processed_boxes % 50 == 0:
                    print(f"  已处理 {processed_boxes}/{total_boxes} 个箱体")
        
        if verbose:
            print(f"特征提取完成，共处理 {processed_boxes} 个箱体")
        
        return detection_results
    
    def extract_from_directory(
        self,
        directory: str,
        extensions: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        从目录中的图像提取特征
        
        Args:
            directory: 图像目录
            extensions: 图像扩展名列表
            verbose: 是否显示进度
            
        Returns:
            {图像路径: 特征向量} 字典
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        directory = Path(directory)
        if not directory.exists():
            print(f"错误: 目录不存在 - {directory}")
            return {}
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        
        if verbose:
            print(f"找到 {len(image_paths)} 张图片")
        
        features_dict = {}
        
        for idx, img_path in enumerate(image_paths):
            if verbose:
                print(f"处理 [{idx+1}/{len(image_paths)}]: {img_path.name}")
            
            try:
                image = Image.open(img_path)
                feature = self.extract_single_image(image)
                features_dict[str(img_path)] = feature
            except Exception as e:
                print(f"警告: 处理失败 - {img_path}: {e}")
        
        return features_dict


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    保存检测结果到文件
    
    Args:
        results: 检测结果列表
        output_path: 输出路径 (.pkl 或 .npy)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        np.save(output_path, results, allow_pickle=True)
    
    print(f"结果已保存到: {output_path}")


def load_results(input_path: str) -> List[Dict[str, Any]]:
    """
    加载检测结果
    
    Args:
        input_path: 输入路径 (.pkl 或 .npy)
        
    Returns:
        检测结果列表
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    if input_path.suffix == '.pkl':
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    else:
        return np.load(input_path, allow_pickle=True).tolist()


def print_feature_summary(results: List[Dict[str, Any]]) -> None:
    """打印特征提取结果摘要"""
    total_images = len(results)
    total_boxes = sum(len(r.get("detections", [])) for r in results)
    boxes_with_features = sum(
        1 for r in results 
        for det in r.get("detections", []) 
        if det.get("feature_vector") is not None
    )
    
    feature_dims = set()
    for r in results:
        for det in r.get("detections", []):
            fv = det.get("feature_vector")
            if fv is not None:
                feature_dims.add(fv.shape[0])
    
    print("\n" + "=" * 60)
    print("特征提取结果摘要")
    print("=" * 60)
    print(f"总图片数: {total_images}")
    print(f"总箱体数: {total_boxes}")
    print(f"成功提取特征的箱体数: {boxes_with_features}")
    print(f"特征维度: {feature_dims if feature_dims else 'N/A'}")
    print("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='特征提取模块 - 使用 ResNet50 提取箱体视觉特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 从检测结果文件提取特征
    python feature_extractor.py --input detections.pkl --output features.pkl
    
    # 从图像目录提取特征
    python feature_extractor.py --images /path/to/images --output features.pkl
    
    # 指定模型和设备
    python feature_extractor.py --input detections.pkl --model resnet101 --device cuda:0
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='检测结果文件路径 (.pkl)'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        help='图像目录路径（直接从图像提取特征）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="feature_results.pkl",
        help='输出文件路径 (默认: feature_results.pkl)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        help='模型名称 (默认: resnet50)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='推理设备 (如 cuda:0, cpu)'
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
    
    extractor = FeatureExtractor(
        model_name=args.model,
        device=args.device
    )
    
    if args.input:
        if verbose:
            print(f"加载检测结果: {args.input}")
        results = load_results(args.input)
        
        results = extractor.extract_from_detections(results, verbose=verbose)
        
        if verbose:
            print_feature_summary(results)
        
        save_results(results, args.output)
        
    elif args.images:
        features_dict = extractor.extract_from_directory(args.images, verbose=verbose)
        
        results = [
            {
                "image_path": path,
                "feature_vector": feature
            }
            for path, feature in features_dict.items()
        ]
        
        save_results(results, args.output)
        
    else:
        print("错误: 请指定 --input 或 --images 参数")
        sys.exit(1)
    
    return results


if __name__ == '__main__':
    main()
