import os
import sys
import argparse
import logging
import yaml
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
