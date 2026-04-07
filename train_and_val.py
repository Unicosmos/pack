"""
训练和验证自定义YOLOv8模型（带CBAM注意力机制）
集成训练和验证功能，参考 pack_train.py 的方式

使用方法：
    # 使用默认配置训练（带CBAM）
    python train_and_val.py --device 0
    
    # 指定参数训练
    python train_and_val.py --epochs 50 --batch 16 --device 0
    
    # 只验证不训练
    python train_and_val.py --val-only --weights runs/detect/yolov8n_cbam_xxx/weights/best.pt
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo

# 先导入自定义模块，注册 CBAM
from register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO

BEIJING_TZ = ZoneInfo('Asia/Shanghai')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_custom.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自定义YOLOv8训练和验证脚本（带CBAM注意力机制）')
    
    # 配置文件
    parser.add_argument('--config', type=str, 
                       default='configs/pack_training_config.yaml',
                       help='训练配置文件路径')
    
    # 训练参数
    parser.add_argument('--name', type=str, default='yolov8n_cbam',
                       help='实验名称')
    parser.add_argument('--model', type=str, default='yolov8n_cbam.yaml',
                       choices=['yolov8n.yaml', 'yolov8n_cbam.yaml', 'yolov8n_p2.yaml', 'yolov8n_p2_cbam.yaml'],
                       help='YOLO模型配置文件（yaml）或预训练权重（pt）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                       help='训练设备，如：0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8,
                       help='数据加载进程数')
    parser.add_argument('--lr0', type=float, default=0.001,
                       help='初始学习率')
    
    # 验证参数
    parser.add_argument('--val-only', action='store_true',
                       help='只进行验证，不训练')
    parser.add_argument('--weights', type=str, default=None,
                       help='验证时使用的权重文件路径')
    
    # 数据集
    parser.add_argument('--data', type=str, 
                       default='/root/source/data2/hyg/projects/hs/mypack/data.yaml',
                       help='数据集配置文件路径')
    
    # CBAM 参数
    parser.add_argument('--use-cbam', action='store_true', default=True,
                       help='使用 CBAM 注意力机制')
    parser.add_argument('--no-cbam', action='store_true',
                       help='不使用 CBAM 注意力机制')
    
    return parser.parse_args()


def load_configuration(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"加载配置文件: {config_path}")
    else:
        config = {}
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
    
    # 用命令行参数覆盖
    config['name'] = args.name
    config['model'] = args.model
    config['epochs'] = args.epochs
    config['batch'] = args.batch
    config['imgsz'] = args.imgsz
    config['workers'] = args.workers
    config['lr0'] = args.lr0
    config['data'] = args.data
    config['use_cbam'] = args.use_cbam and not args.no_cbam
    
    # 处理设备参数
    if args.device.lower() == 'cpu':
        config['device'] = 'cpu'
    else:
        device_list = [int(d.strip()) for d in args.device.split(',')]
        config['device'] = device_list
    
    # 添加时间戳到名称
    origin_name = config['name']
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
    config['name'] = f"{origin_name}_{timestamp}"
    config['experiment_name'] = origin_name
    logger.info(f"实验名称: {config['name']}")
    
    # 设置其他默认参数（与原配置保持一致）
    config.setdefault('project', '/root/source/data2/hyg/projects/hs/runs/detect')
    config.setdefault('exist_ok', False)
    config.setdefault('pretrained', True)
    config.setdefault('optimizer', 'auto')
    config.setdefault('cos_lr', True)
    config.setdefault('patience', 10)
    config.setdefault('save', True)
    config.setdefault('save_period', -1)
    config.setdefault('cache', True)
    config.setdefault('amp', True)
    config.setdefault('verbose', True)
    config.setdefault('val', True)
    config.setdefault('plots', True)
    config.setdefault('mosaic', 0.6)
    config.setdefault('mixup', 0.1)
    config.setdefault('hsv_h', 0.015)
    config.setdefault('hsv_s', 0.7)
    config.setdefault('hsv_v', 0.4)
    config.setdefault('degrees', 0.0)
    config.setdefault('translate', 0.1)
    config.setdefault('scale', 0.5)
    config.setdefault('fliplr', 0.5)
    config.setdefault('warmup_epochs', 3)
    config.setdefault('weight_decay', 0.0005)
    
    return config


def train(config):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    
    model_yaml = config['model']
    
    if 'p2' in model_yaml.lower():
        logger.info("使用 P2 检测层 (小目标优化)")
    if 'cbam' in model_yaml.lower():
        logger.info("使用 CBAM 注意力机制")
    
    logger.info(f"模型配置: {model_yaml}")
    model = YOLO(model_yaml)
    
    # 准备训练参数
    train_params = {}
    excluded_keys = ['model', 'data', 'mlflow-uri', 'experiment_name', 'use_cbam']
    
    for key, value in config.items():
        if key in excluded_keys:
            continue
        train_params[key] = value
    
    logger.info(f"数据集: {config['data']}")
    logger.info(f"设备: {config['device']}")
    logger.info(f"训练轮数: {config['epochs']}")
    logger.info(f"批次大小: {config['batch']}")
    logger.info(f"图像尺寸: {config['imgsz']}")
    logger.info(f"CBAM: {'启用' if config['use_cbam'] else '禁用'}")
    
    # 开始训练
    results = model.train(
        data=config['data'],
        **train_params
    )
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳模型: runs/detect/{config['name']}/weights/best.pt")
    logger.info(f"最后模型: runs/detect/{config['name']}/weights/last.pt")
    logger.info("=" * 60)
    
    return model, results


def validate(weights_path, data_path, device='0', imgsz=640, batch=16):
    """验证模型"""
    logger.info("=" * 60)
    logger.info("开始验证")
    logger.info("=" * 60)
    
    # 加载训练好的模型
    logger.info(f"加载模型权重: {weights_path}")
    model = YOLO(weights_path)
    
    # 验证
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True
    )
    
    # 打印结果
    logger.info("=" * 60)
    logger.info("验证结果:")
    logger.info("=" * 60)
    logger.info(f"mAP50:    {results.box.map50:.4f}")
    logger.info(f"mAP50-95: {results.box.map:.4f}")
    logger.info(f"精确率:   {results.box.mp:.4f}")
    logger.info(f"召回率:   {results.box.mr:.4f}")
    logger.info("=" * 60)
    
    return results


def main():
    """主函数"""
    args = parse_arguments()
    config = load_configuration(args.config, args)
    
    if args.val_only:
        # 只验证模式
        if args.weights is None:
            logger.error("验证模式需要指定 --weights 参数")
            sys.exit(1)
        validate(args.weights, config['data'], config['device'], config['imgsz'], config['batch'])
    else:
        # 训练模式
        train(config)


if __name__ == '__main__':
    main()
