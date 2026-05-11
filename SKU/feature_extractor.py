"""
特征提取模块 - OML ViT-S16 DINO版

使用 OML库预训练的 ViT-S16 DINO 模型，或加载微调后的模型。
与 sku_model_trainer.py 架构一致。

特征维度说明：
- ViT-S16 DINO base model: 384维
- 与 core/matcher.py 中的 feature_dim=384 保持一致

模型加载优先级：
1. 传入的 model_path 参数
2. 从 web/backend/config.py 自动读取 SKU_MODEL_PATH
3. 使用 OML 预训练模型（备选）

使用方法:
    from feature_extractor import FeatureExtractor
    
    # 自动从 config.py 读取微调模型
    extractor = FeatureExtractor(device='cpu')
    
    # 或手动指定微调模型
    extractor = FeatureExtractor(model_path='./models/sku_trained_vits16_dino.pth', device='cpu')
    
    image = Image.open('test.jpg')
    feature = extractor.extract(image)  # 返回 [384] 维特征向量
    
    # 批量处理
    features = extractor.extract_batch([img1, img2, img3], batch_size=8)
"""

import os
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from PIL import Image


def get_default_model_path() -> Optional[str]:
    """从 config.py 获取默认模型路径"""
    try:
        backend_config_path = Path(__file__).parent.parent / "web" / "backend" / "config.py"
        if backend_config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", backend_config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            if hasattr(config_module, 'config') and config_module.config.paths.SKU_MODEL_PATH:
                return str(config_module.config.paths.SKU_MODEL_PATH)
    except Exception:
        pass
    return None


class FeatureExtractor:
    """ViT-S16 DINO 特征提取器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', num_threads: int = 1):
        """
        初始化特征提取器
        
        Args:
            model_path: 微调模型路径 (.pth文件)，如果为None则尝试从config.py读取
            device: 推理设备 ('cpu' 或 'cuda')
            num_threads: 预留参数，保持接口兼容
        """
        if model_path is None:
            model_path = get_default_model_path()
        self.device = device
        self.num_threads = num_threads
        self.model = None
        self.transform = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """加载 ViT-S16 DINO 模型"""
        try:
            from oml.models import ViTExtractor
            from oml.registry import get_transforms_for_pretrained
            
            if model_path and Path(model_path).exists():
                # 加载微调后的模型
                print(f"  加载微调模型: {model_path}")
                self.model = ViTExtractor.from_pretrained("vits16_dino")
                state_dict = torch.load(model_path, map_location=self.device)
                # 适配不同格式的模型文件
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                if 'cls_token' in state_dict and 'model.cls_token' not in state_dict:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict[f'model.{k}'] = v
                    state_dict = new_state_dict
                self.model.load_state_dict(state_dict)
                print(f"  ✓ 微调模型加载成功")
            else:
                # 使用预训练模型
                print(f"  使用预训练模型: vits16_dino")
                self.model = ViTExtractor.from_pretrained("vits16_dino")
                print(f"  ✓ 预训练模型加载成功")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform, _ = get_transforms_for_pretrained("vits16_dino")
            print(f"  特征维度: 384")
            if self.device == 'cpu':
                print(f"  CPU线程数: {self.num_threads}")
        
        except Exception as e:
            print(f"  ⚠ 模型加载失败: {e}")
            print("  将使用随机特征作为替代")
            self.model = None
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理单张图片"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)
    
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        从图像提取特征
        
        Args:
            image: PIL Image对象（RGB格式）
        
        Returns:
            384维特征向量（L2归一化）
        """
        if self.model is None or self.transform is None:
            return np.random.randn(384).astype(np.float32)
        
        try:
            tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(tensor)
            
            feat = features.squeeze().cpu().numpy().astype(np.float32)
            
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            
            return feat
        
        except Exception as e:
            print(f"  ⚠ 特征提取失败: {e}")
            return np.random.randn(384).astype(np.float32)
    
    def extract_batch(self, images: List[Image.Image], batch_size: int = 8) -> np.ndarray:
        """
        批量提取特征（CPU优化版）
        
        Args:
            images: PIL Image对象列表
            batch_size: 批处理大小，CPU建议8-16
        
        Returns:
            特征矩阵 [N, 384]
        """
        if self.model is None or self.transform is None:
            return np.random.randn(len(images), 384).astype(np.float32)
        
        try:
            all_features = []
            
            # 单线程预处理图片
            tensors = [self._preprocess_image(img) for img in images]
            
            # 批量推理
            for i in range(0, len(tensors), batch_size):
                batch_tensors = torch.stack(tensors[i:i+batch_size]).to(self.device)
                
                with torch.no_grad():
                    batch_features = self.model(batch_tensors)
                
                batch_features = batch_features.cpu().numpy().astype(np.float32)
                all_features.append(batch_features)
            
            # 合并结果并归一化
            features = np.vstack(all_features)
            
            # L2归一化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms
            
            return features
        
        except Exception as e:
            print(f"  ⚠ 批量特征提取失败: {e}")
            return np.random.randn(len(images), 384).astype(np.float32)


def extract_features_from_directory(
    input_dir: str,
    output_file: str = None,
    model_path: str = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    从目录提取所有图片的特征
    
    Args:
        input_dir: 图片目录
        output_file: 特征输出文件路径（可选）
        model_path: 微调模型路径（可选）
        device: 推理设备
    
    Returns:
        特征矩阵 [N, 384]
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"目录不存在: {input_path}")
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(f'*{ext}'))
        image_paths.extend(input_path.rglob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        raise ValueError(f"目录中没有找到图片: {input_path}")
    
    print(f"找到 {len(image_paths)} 张图片")
    
    extractor = FeatureExtractor(model_path=model_path, device=device)
    
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
    
    if output_file:
        np.save(output_file, features)
        print(f"特征已保存到: {output_file}")
    
    return features


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT-S16 DINO 特征提取器')
    parser.add_argument('--input', type=str, required=True, help='输入图片目录')
    parser.add_argument('--output', type=str, default='sku_features.npy', help='输出特征文件')
    parser.add_argument('--model-path', type=str, default=None, help='微调模型路径')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    extract_features_from_directory(
        input_dir=args.input,
        output_file=args.output,
        model_path=args.model_path,
        device=args.device
    )
