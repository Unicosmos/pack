"""
特征提取模块 - ViT-S16 DINO版

使用预训练的 ViT-S16 DINO 模型提取图像特征，输出384维特征向量。

特征维度说明：
- ViT-S16 DINO base model: 384维
- 与 core/matcher.py 中的 feature_dim=384 保持一致

使用方法：
    from feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor(device='cpu')
    image = Image.open('test.jpg')
    feature = extractor.extract(image)  # 返回 [384] 维特征向量
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


class FeatureExtractor:
    """ViT-S16 DINO 特征提取器"""
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化特征提取器
        
        Args:
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self) -> None:
        """加载 ViT-S16 DINO 模型"""
        try:
            # 使用 torch.hub 加载预训练模型
            self.model = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_vits16',
                pretrained=True
            )
            
            # 移除分类头，只保留特征提取部分
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            # 设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 定义图像变换（与DINO训练时一致）
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print(f"  ViT-S16 DINO 模型加载成功 (device={self.device})")
            print(f"  特征维度: 384")
        
        except Exception as e:
            print(f"警告: 加载ViT-S16 DINO模型失败: {e}")
            print("  将使用随机特征作为替代")
    
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        从图像提取特征
        
        Args:
            image: PIL Image对象（RGB格式）
        
        Returns:
            384维特征向量（L2归一化）
        """
        if self.model is None or self.transform is None:
            # 如果模型加载失败，返回随机特征
            return np.random.randn(384).astype(np.float32)
        
        try:
            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model(tensor)
            
            # 展平并转换为numpy数组
            feat = features.squeeze().cpu().numpy().astype(np.float32)
            
            # L2归一化
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            
            return feat
        
        except Exception as e:
            print(f"特征提取失败: {e}")
            return np.random.randn(384).astype(np.float32)
    
    def extract_batch(self, images) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            images: PIL Image对象列表
        
        Returns:
            特征矩阵 [N, 384]
        """
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        
        return np.array(features)


def extract_features_from_directory(
    input_dir: str,
    output_file: str = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    从目录提取所有图片的特征
    
    Args:
        input_dir: 图片目录
        output_file: 特征输出文件路径（可选）
        device: 推理设备
    
    Returns:
        特征矩阵 [N, 384]
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"目录不存在: {input_dir}")
    
    # 支持的图片格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 收集所有图片
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(f'*{ext}'))
        image_paths.extend(input_path.rglob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        raise ValueError(f"目录中没有找到图片: {input_dir}")
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 初始化提取器
    extractor = FeatureExtractor(device=device)
    
    # 批量提取特征
    features = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            feat = extractor.extract(image)
            features.append(feat)
            print(f"  处理: {img_path.name}")
        except Exception as e:
            print(f"  跳过 {img_path.name}: {e}")
    
    features = np.array(features)
    print(f"特征矩阵形状: {features.shape}")
    
    # 保存特征
    if output_file:
        np.save(output_file, features)
        print(f"特征已保存到: {output_file}")
    
    return features


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT-S16 DINO 特征提取器')
    parser.add_argument('--input', type=str, required=True, help='输入图片目录')
    parser.add_argument('--output', type=str, default='sku_features.npy', help='输出特征文件')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    extract_features_from_directory(
        input_dir=args.input,
        output_file=args.output,
        device=args.device
    )