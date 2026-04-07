import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pack_val.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO包装检测验证脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, 
                       default='configs/pack_val_config.yaml',
                       help='验证配置文件路径')
    
    # 基础配置
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO模型权重文件路径')
    parser.add_argument('--data', type=str, default=None,
                       help='数据集配置文件路径')
    
    # 验证参数
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--conf', type=float, default=None,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=None,
                       help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default=None,
                       help='验证设备，如：0,1,2,3 或 cpu')
    parser.add_argument('--half', action='store_true',
                       help='使用FP16半精度推理')
    
    # 输出配置
    parser.add_argument('--project', type=str, default=None,
                       help='保存结果的项目目录')
    parser.add_argument('--save-json', action='store_true',
                       help='保存结果到JSON文件')
    parser.add_argument('--plots', action='store_true',
                       help='生成验证结果图表')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细验证日志')
    
    return parser.parse_args()


def load_configuration(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    config = {}
    
    # 如果提供了配置文件，则加载它
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"加载配置文件: {config_path}")
    else:
        logger.warning(f"配置文件不存在: {config_path}，将使用命令行参数")
    
    # 用命令行参数覆盖配置文件中的设置
    if args.model is not None:
        config['model'] = args.model
        logger.info(f"命令行覆盖模型路径: {args.model}")
    
    if args.data is not None:
        config['data'] = args.data
        logger.info(f"命令行覆盖数据集路径: {args.data}")
    
    # 设置实验名称，自动添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.name is not None:
        config['name'] = f"{args.name}_{timestamp}"
        logger.info(f"命令行设置实验名称: {config['name']}")
    elif 'name' in config and config['name']:
        config['name'] = f"{config['name']}_{timestamp}"
        logger.info(f"配置文件实验名称，添加时间戳: {config['name']}")
    else:
        config['name'] = f"val_{timestamp}"
        logger.info(f"设置默认实验名称: {config['name']}")
    
    # 验证参数
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
        logger.info(f"命令行覆盖图像尺寸: {args.imgsz}")
    
    if args.batch is not None:
        config['batch'] = args.batch
        logger.info(f"命令行覆盖批次大小: {args.batch}")
    
    if args.conf is not None:
        config['conf'] = args.conf
        logger.info(f"命令行覆盖置信度阈值: {args.conf}")
    
    if args.iou is not None:
        config['iou'] = args.iou
        logger.info(f"命令行覆盖IoU阈值: {args.iou}")
    
    if args.device is not None:
        config['device'] = args.device
        logger.info(f"命令行覆盖设备: {args.device}")
    
    if args.half:
        config['half'] = True
        logger.info("启用FP16半精度推理")
    
    # 输出配置
    if args.project is not None:
        config['project'] = args.project
        logger.info(f"命令行覆盖项目目录: {args.project}")
    
    if args.save_json:
        config['save_json'] = True
        logger.info("启用JSON保存")
    
    if args.plots:
        config['plots'] = True
        logger.info("启用图表生成")
    
    if args.verbose:
        config['verbose'] = True
        logger.info("启用详细日志")
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_configuration(args.config, args)
    
    # 检查必需参数
    if 'model' not in config or not config['model']:
        logger.error("错误：必须指定模型路径（通过 --model 参数或配置文件）")
        sys.exit(1)
    
    if 'data' not in config or not config['data']:
        logger.error("错误：必须指定数据集路径（通过 --data 参数或配置文件）")
        sys.exit(1)
    
    # 加载模型
    logger.info(f"加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 确定任务类型
    task_type = model.task
    logger.info(f"模型任务类型: {task_type}")
    
    # 准备验证参数
    val_args = {}
    excluded_keys = ['model', 'data']
    
    for key, value in config.items():
        if key in excluded_keys:
            continue
        val_args[key] = value
    
    try:
        logger.info("开始验证...")
        logger.info(f"模型: {config['model']}")
        logger.info(f"数据集: {config['data']}")
        
        # 执行验证
        results = model.val(data=config['data'], **val_args)
        
        # 打印验证结果摘要
        logger.info("=== 验证结果摘要 ===")
        
        # 获取主要指标
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            
            # 打印mAP指标
            if 'metrics/mAP50(B)' in metrics:
                logger.info(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                logger.info(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
            
            # 打印精度和召回率
            if 'metrics/precision(B)' in metrics:
                logger.info(f"Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                logger.info(f"Recall: {metrics['metrics/recall(B)']:.4f}")
            
            # 打印F1分数
            if 'metrics/precision(B)' in metrics and 'metrics/recall(B)' in metrics:
                precision = metrics['metrics/precision(B)']
                recall = metrics['metrics/recall(B)']
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                logger.info(f"F1 Score: {f1:.4f}")
        
        # 打印每个类别的指标
        if hasattr(results, 'names') and hasattr(results, 'curves'):
            logger.info("\n=== 各类别指标 ===")
            for i, name in results.names.items():
                logger.info(f"类别 {i} ({name}):")
                # 这里可以添加更详细的类别指标
        
        logger.info("\n验证完成！")
        
        # 显示结果保存路径
        if hasattr(results, 'save_dir'):
            logger.info(f"验证结果已保存到: {results.save_dir}")
        
    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        import traceback
        logger.error(f"错误堆栈跟踪:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
