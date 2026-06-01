"""
匹配服务 - 封装SKU特征匹配功能
"""

from typing import Optional, List, Dict, Any
from PIL import Image

from core.matcher import SKUMatcher
from core.utils.image_utils import resize_with_padding
from core.utils.logger import logger
from config import config


class MatchService:
    """匹配服务类"""
    
    def __init__(self):
        self.matcher: Optional[SKUMatcher] = None
        self._ready = False
    
    def initialize(self):
        """初始化匹配器"""
        cfg = config
        
        if cfg.paths.SKU_DIR.exists():
            logger.info(f"加载SKU库: {cfg.paths.SKU_DIR}")
            try:
                self.matcher = SKUMatcher(
                    str(cfg.paths.SKU_DIR),
                    match_threshold=cfg.match.MATCH_THRESHOLD,
                    sku_model_path=str(cfg.paths.SKU_MODEL_PATH) if cfg.paths.SKU_MODEL_PATH else None
                )
                if self.matcher.is_ready():
                    self._ready = True
                    logger.info("  SKUMatcher加载成功")
                else:
                    logger.warning("  SKUMatcher未就绪（可能缺少特征文件）")
            except Exception as e:
                logger.error(f"  SKUMatcher加载失败: {e}")
        else:
            logger.info("  SKU库目录不存在，匹配功能将不可用")
    
    def is_ready(self) -> bool:
        """检查匹配器是否就绪"""
        return self._ready and self.matcher is not None
    
    def match_single(self, image: Image.Image, match_threshold: float = None) -> Dict[str, Any]:
        """
        对单张图片进行SKU匹配
        
        Args:
            image: PIL Image对象
            match_threshold: 相似度阈值
            
        Returns:
            匹配结果字典
        """
        if not self.is_ready():
            raise RuntimeError("SKU匹配器未加载")
        
        resized = resize_with_padding(image, target_size=config.model.INPUT_SIZE)
        features = self.matcher.extract_feature(resized)
        if match_threshold is None:
            match_threshold = config.match.MATCH_THRESHOLD
        result = self.matcher.match_sku(features, threshold=match_threshold)
        
        return {
            "sku_id": result.sku_id,
            "sku_name": result.sku_name,
            "similarity": result.similarity,
            "ratio": result.ratio,
            "status": result.status,
            "top5_labels": result.top5_labels
        }
    
    def match_batch(self, image: Image.Image, boxes: List[Dict], match_threshold: float = None) -> List[Dict]:
        """
        批量匹配SKU
        
        Args:
            image: PIL Image对象
            boxes: 检测框列表
            match_threshold: 相似度阈值
            
        Returns:
            匹配结果列表
        """
        if not self.is_ready():
            return [None] * len(boxes)
        
        try:
            if match_threshold is None:
                match_threshold = config.match.MATCH_THRESHOLD
            match_results = []
            for box in boxes:
                bbox = box.get("bbox", [])
                cropped = crop_box(image, bbox) if hasattr(box, 'get') else None
                if cropped:
                    resized = resize_with_padding(cropped, target_size=config.model.INPUT_SIZE)
                    feat = self.matcher.extract_feature(resized)
                    mr = self.matcher.match_sku(feat, threshold=match_threshold)
                    
                    match_results.append({
                        "sku_id": mr.sku_id,
                        "sku_name": mr.sku_name,
                        "similarity": mr.similarity,
                        "ratio": mr.ratio,
                        "status": mr.status,
                        "top5_labels": mr.top5_labels
                    })
                else:
                    match_results.append(None)
            
            return match_results
        except Exception as e:
            logger.error(f"批量匹配失败: {e}")
            return [None] * len(boxes)
    
    def get_sku_count(self) -> int:
        """获取SKU数量"""
        if self.is_ready():
            sku_ids = set()
            for item in self.matcher.sku_info:
                sku_ids.add(item.get('sku_id', ''))
            return len(sku_ids)
        return 0
    
    def get_sku_list(self) -> List[Dict]:
        """获取SKU列表"""
        if not self.is_ready():
            return []
        
        sku_map = {}
        for item in self.matcher.sku_info:
            sku_id = item.get('sku_id', '')
            if sku_id:
                if sku_id not in sku_map:
                    sku_map[sku_id] = {
                        'sku_id': sku_id,
                        'sku_name': item.get('sku_name', sku_id),
                        'labels': []
                    }
                sku_map[sku_id]['labels'].append(item.get('label', ''))
        
        return [
            {
                'sku_id': sku_id,
                'sku_name': info['sku_name'],
                'label_count': len(info['labels']),
                'image_count': len(info['labels'])
            }
            for sku_id, info in sku_map.items()
        ]


def crop_box(image: Image.Image, bbox: List[int], expand_ratio: float = 0.0):
    """裁剪图片（内部辅助函数）"""
    from core.utils.image_utils import crop_box as _crop_box
    return _crop_box(image, bbox, expand_ratio)
