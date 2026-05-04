import os
import sys
import argparse
import logging
import yaml

# ============ 【新增】遮挡增强需要的导入 ============
import random
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
# ============ 新增结束 ============

from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO, settings
from dotenv import load_dotenv

BEIJING_TZ = ZoneInfo('Asia/Shanghai')

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pack_train.log')
    ]
)
logger = logging.getLogger(__name__)


# ============ 【新增】遮挡增强函数 ============

OCCLUSION_AUG_CONFIG = {
    'aug_prob': 1.0,           # 每个样本增强概率
    'target_classes': [1, 3],  # 目标类别（两个occlusion类别）
    'num_holes_range': (1, 5), # 挖洞数量
    'hole_size_range': (10, 40), # 洞大小（像素）
}


def parse_yolo_seg_label(label_path: str, img_h: int, img_w: int):
    """解析YOLO分割标注（多边形格式）"""
    if not os.path.exists(label_path):
        return [], [], []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    classes, polygons, masks = [], [], []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + 至少3个点
            continue
        
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                points.append([x, y])
            
            polygon = np.array(points, dtype=np.float32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
            
            classes.append(cls)
            polygons.append(polygon)
            masks.append(mask)
        except (ValueError, IndexError):
            continue
    
    return classes, polygons, masks


def mask_to_polygon(mask: np.ndarray):
    """从mask提取多边形轮廓"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    return approx.reshape(-1, 2).astype(np.float32)


def apply_occlusion_augmentation(image_path, label_path, output_image_path, output_label_path, config=None):
    """对单张分割图像应用遮挡增强"""
    cfg = {**OCCLUSION_AUG_CONFIG, **(config or {})}
    
    if random.random() > cfg['aug_prob']:
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    H, W = image.shape[:2]
    classes, polygons, masks = parse_yolo_seg_label(label_path, H, W)
    
    if not classes:
        return False
    
    target_indices = [i for i, c in enumerate(classes) if c in cfg['target_classes']]
    if not target_indices:
        return False
    
    image = image.copy()
    augmented = False
    
    for idx in target_indices:
        mask = masks[idx]
        if mask.sum() == 0:
            continue
        
        # 找边界
        kernel = np.ones((5, 5), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        boundary_coords = np.where(boundary > 0)
        
        if len(boundary_coords[0]) == 0:
            continue
        
        # 挖洞
        num_holes = random.randint(*cfg['num_holes_range'])
        for _ in range(num_holes):
            point_idx = random.randint(0, len(boundary_coords[0]) - 1)
            cy, cx = boundary_coords[0][point_idx], boundary_coords[1][point_idx]
            hole_size = random.randint(*cfg['hole_size_range'])
            
            y1 = max(0, cy - hole_size // 2)
            x1 = max(0, cx - hole_size // 2)
            y2 = min(H, y1 + hole_size)
            x2 = min(W, x1 + hole_size)
            
            fill_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            image[y1:y2, x1:x2] = fill_color
            masks[idx][y1:y2, x1:x2] = 0
        
        augmented = True
    
    if not augmented:
        return False
    
    # 保存增强后的标注
    with open(output_label_path, 'w') as f:
        for cls, mask in zip(classes, masks):
            if mask.sum() == 0:
                continue
            new_polygon = mask_to_polygon(mask)
            if new_polygon is None or len(new_polygon) < 3:
                continue
            
            coords = []
            for x, y in new_polygon:
                coords.extend([x / W, y / H])
            coords_str = ' '.join([f'{c:.6f}' for c in coords])
            f.write(f"{cls} {coords_str}\n")
    
    cv2.imwrite(output_image_path, image)
    return True


def preprocess_dataset_with_occlusion(source_yaml, output_dir, aug_ratio=0.3,seed=0):
    """预处理数据集：生成遮挡增强版本"""
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(source_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    source_path = Path(data_config['path'])
    train_rel_path = data_config['train']
    
    # 创建目录
    out_train_images = output_dir / 'images' / 'train'
    out_train_labels = output_dir / 'labels' / 'train'
    out_val_images = output_dir / 'images' / 'val'
    out_val_labels = output_dir / 'labels' / 'val'
    
    for d in [out_train_images, out_train_labels, out_val_images, out_val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 处理训练集
    train_images_dir = source_path / 'images' / 'train'
    train_labels_dir = source_path / 'labels' / 'train'
    image_files = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
    
    logger.info("=" * 60)
    logger.info("开始预处理数据集（遮挡增强）")
    logger.info(f"原始数据集: {source_yaml}")
    logger.info(f"训练集图像数: {len(image_files)}")
    logger.info(f"增强比例: {aug_ratio}")
    logger.info("=" * 60)
    
    aug_count = 0
    for img_file in tqdm(image_files, desc="预处理"):
        # 复制原始
        shutil.copy(img_file, out_train_images / img_file.name)
        label_file = train_labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(label_file, out_train_labels / label_file.name)
        
        # 创建增强版
        if random.random() < aug_ratio:
            aug_img = out_train_images / f"{img_file.stem}_aug{img_file.suffix}"
            aug_label = out_train_labels / f"{img_file.stem}_aug.txt"
            try:
                if apply_occlusion_augmentation(str(img_file), str(label_file), str(aug_img), str(aug_label)):
                    aug_count += 1
            except Exception as e:
                logger.warning(f"处理失败: {img_file.name}, {e}")
    
    # 复制验证集
    if 'val' in data_config:
        val_images_dir = source_path / 'images' / 'val'
        val_labels_dir = source_path / 'labels' / 'val'
        if val_images_dir.exists():
            val_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
            for img_file in tqdm(val_files, desc="复制验证集"):
                shutil.copy(img_file, out_val_images / img_file.name)
                label_file = val_labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    shutil.copy(label_file, out_val_labels / label_file.name)
    
    # 生成配置
    output_config = data_config.copy()
    output_config['path'] = str(output_dir.absolute())
    output_config['train'] = 'images/train'
    output_config['val'] = 'images/val'
    
    output_yaml = output_dir / 'dataset.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(output_config, f)
    
    logger.info(f"预处理完成！增强样本数: {aug_count}")
    logger.info(f"输出配置: {output_yaml}")
    
    return str(output_yaml)

# ============ 新增函数结束 ============


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO包装检测训练脚本')
    
    parser.add_argument('--config', type=str, 
                       default='configs/hyp_lscd.yaml',
                       help='训练配置文件路径')
    
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称，覆盖配置文件中的设置')
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO基础模型路径，覆盖配置文件中的设置')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数，覆盖配置文件中的设置')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小，覆盖配置文件中的设置')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸，覆盖配置文件中的设置')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备，如：0,1,2,3 或 cpu，覆盖配置文件中的设置')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载进程数，覆盖配置文件中的设置')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率，覆盖配置文件中的设置')
    
    parser.add_argument('--profile', action='store_true',
                       help='启用性能分析')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细训练日志')
    parser.add_argument('--val', action='store_true',
                       help='启用验证')
    parser.add_argument('--plots', action='store_true',
                       help='生成训练图表')
    
    # ============ 【新增】遮挡增强参数 ============
    parser.add_argument('--occlusion-aug', action='store_true',
                       help='启用遮挡增强预处理')
    parser.add_argument('--aug-ratio', type=float, default=0.3,
                       help='遮挡增强比例（默认0.3）')
    # ============ 新增参数结束 ============
    
    return parser.parse_args()

def load_configuration(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    
    if args.name is not None:
        config['name'] = args.name
        logger.info(f"覆盖实验名称: {args.name}")
    
    origin_name = config['name']
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
    config['experiment_name'] = origin_name
    config['name'] = f"{origin_name}_{timestamp}"
    logger.info(f"本地文件夹名称: {config['name']} (实验名称: {origin_name})")
    
    if args.model is not None:
        config['model'] = args.model
        logger.info(f"覆盖模型路径: {args.model}")
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
        logger.info(f"覆盖训练轮数: {args.epochs}")
    
    if args.batch is not None:
        config['batch'] = args.batch
        logger.info(f"覆盖批次大小: {args.batch}")
    
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
        logger.info(f"覆盖图像尺寸: {args.imgsz}")
    
    if args.device is not None:
        # 处理设备参数
        if args.device.lower() == 'cpu':
            config['device'] = 'cpu'
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(',')]
            config['device'] = device_list
        logger.info(f"覆盖设备: {config['device']}")
    
    if args.workers is not None:
        config['workers'] = args.workers
        logger.info(f"覆盖数据加载进程数: {args.workers}")
    
    if args.lr0 is not None:
        config['lr0'] = args.lr0
        logger.info(f"覆盖初始学习率: {args.lr0}")
    
    if args.profile:
        config['profile'] = True
        logger.info("启用性能分析")
    
    if args.verbose:
        config['verbose'] = True
        logger.info("启用详细日志")
    
    if args.val:
        config['val'] = True
        logger.info("启用验证")
    
    if args.plots:
        config['plots'] = True
        logger.info("生成训练图表")
    
    return config

def setup_mlflow_integration(config):
    """配置MLflow集成"""
    if 'mlflow-uri' in config and config['mlflow-uri']:
        mlflow_experiment_name = config.get('experiment_name', config['name'])
        mlflow_run_name = f'{mlflow_experiment_name}_{datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")}'
        
        os.environ['MLFLOW_TRACKING_URI'] = config['mlflow-uri']
        os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
        os.environ['MLFLOW_RUN'] = mlflow_run_name
        
        settings.update({"mlflow": True})
        logger.info(f"MLflow集成已启用 - 服务器: {config['mlflow-uri']}")
        logger.info(f"MLflow实验名称: {mlflow_experiment_name}")
        logger.info(f"MLflow运行名称: {mlflow_run_name}")
    else:
        logger.info("MLflow集成已禁用")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_configuration(args.config, args)
    
    # ============ 【新增】遮挡增强预处理 ============
    if args.occlusion_aug:
        logger.info("启用遮挡增强预处理")
        source_yaml = config['data']
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
        output_dir = f"./augmented_datasets/occlusion_aug_{timestamp}"
        
        aug_yaml = preprocess_dataset_with_occlusion(
            source_yaml, output_dir, aug_ratio=args.aug_ratio
        )
        config['data'] = aug_yaml
        logger.info(f"使用增强数据集: {aug_yaml}")
    # ============ 新增逻辑结束 ============
    
    # 配置MLflow集成
    setup_mlflow_integration(config)
    
    # 加载模型
    logger.info(f"加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 准备训练参数
    train_params = {}
    excluded_keys = ['model', 'data', 'mlflow-uri', 'experiment_name']
    
    for key, value in config.items():
        if key in excluded_keys:
            continue
        train_params[key] = value
    
    try:
        logger.info("开始训练...")
        logger.info(f"使用配置文件: {args.config}")
        logger.info(f"模型: {config['model']}")
        logger.info(f"数据集: {config['data']}")
        
        # 开始训练
        results = model.train(
            data=config['data'],
            **train_params
        )
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(f"错误堆栈跟踪:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()
