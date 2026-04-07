import os
import sys
import argparse
import logging
import yaml
import json
import gc
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pack_predict.log')
    ]
)
logger = logging.getLogger(__name__)


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_single_result(result, save_dir, result_index):
    """处理单个预测结果并保存JSON文件
    
    Args:
        result: YOLO预测结果对象
        save_dir: 保存目录路径
        result_index: 结果索引
    
    Returns:
        bool: 是否成功保存
    """
    try:
        # 获取检测结果
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i in range(len(result.boxes)):
                detection = {}
                
                # 获取置信度
                if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                    detection['confidence'] = result.boxes.conf[i].item()
                
                # 获取类别ID
                if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                    detection['class_id'] = int(result.boxes.cls[i].item())
                
                # 获取类别名称
                if hasattr(result, 'names') and result.names:
                    detection['name'] = result.names.get(detection.get('class_id', 0), 'object')
                
                # 获取边界框坐标
                if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                    box = result.boxes.xyxy[i].cpu().numpy()
                    detection['box'] = {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                
                detections.append(detection)
        
        # 确定文件名
        if hasattr(result, 'path') and result.path:
            source_name = Path(result.path).stem
            json_file = save_dir / f"{source_name}.json"
        else:
            json_file = save_dir / f"result_{result_index}.json"
        
        # 获取图片尺寸
        original_width = result.orig_shape[1] if hasattr(result, 'orig_shape') else 640
        original_height = result.orig_shape[0] if hasattr(result, 'orig_shape') else 640
        
        # 构建结果数据
        result_data = {
            'image_path': str(result.path) if hasattr(result, 'path') else f"result_{result_index}.jpg",
            'image_width': original_width,
            'image_height': original_height,
            'detections': detections,
            'total_detections': len(detections)
        }
        
        # 保存JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        logger.error(f"保存第{result_index}个结果时出错: {e}")
        return False


def process_single_result_yolo(result, save_dir, result_index):
    """处理单个预测结果并保存YOLO格式标注文件
    
    Args:
        result: YOLO预测结果对象
        save_dir: 保存目录路径
        result_index: 结果索引
    
    Returns:
        bool: 是否成功保存
    """
    try:
        # 确定文件名
        if hasattr(result, 'path') and result.path:
            source_name = Path(result.path).stem
            txt_file = save_dir / f"{source_name}.txt"
        else:
            txt_file = save_dir / f"result_{result_index}.txt"
        
        # 获取图片尺寸
        original_width = result.orig_shape[1] if hasattr(result, 'orig_shape') else 640
        original_height = result.orig_shape[0] if hasattr(result, 'orig_shape') else 640
        
        # 准备YOLO格式标注
        yolo_lines = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i in range(len(result.boxes)):
                # 获取类别ID
                if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                    class_id = int(result.boxes.cls[i].item())
                else:
                    continue
                
                # 获取边界框坐标（xyxy格式）
                if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                    box = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # 转换为YOLO格式（中心点坐标+宽高，归一化到0-1）
                    x_center = (x1 + x2) / 2 / original_width
                    y_center = (y1 + y2) / 2 / original_height
                    width = (x2 - x1) / original_width
                    height = (y2 - y1) / original_height
                    
                    # 格式：class_id x_center y_center width height
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 保存YOLO格式文件
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        return True
    
    except Exception as e:
        logger.error(f"保存第{result_index}个YOLO标注文件时出错: {e}")
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO包装检测预测脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, 
                       default='configs/pack_predict_config.yaml',
                       help='预测配置文件路径')
    
    # 基础配置
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO模型权重文件路径')
    parser.add_argument('--source', type=str, default=None,
                       help='输入源：图片路径、目录、视频等')
    
    # 预测参数
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=None,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=None,
                       help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=None,
                       help='每张图像最大检测数')
    parser.add_argument('--device', type=str, default=None,
                       help='推理设备，如：0,1,2,3 或 cpu')
    parser.add_argument('--half', action='store_true',
                       help='使用FP16半精度推理')
    
    # 输出配置
    parser.add_argument('--save', action='store_true',
                       help='保存预测结果图像')
    parser.add_argument('--save-json', action='store_true',
                       help='保存结果到JSON文件')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存结果为YOLO格式标注文件')
    parser.add_argument('--project', type=str, default=None,
                       help='保存结果的项目目录')
    parser.add_argument('--exist-ok', action='store_true',
                       help='允许覆盖已存在的实验目录')
    
    # 显示配置
    parser.add_argument('--show', action='store_true',
                       help='显示预测结果')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细预测日志')
    
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
    
    if args.source is not None:
        config['source'] = args.source
        logger.info(f"命令行覆盖输入源: {args.source}")
    
    # 设置实验名称，自动添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.name is not None:
        config['name'] = f"{args.name}_{timestamp}"
        logger.info(f"命令行设置实验名称: {config['name']}")
    elif 'name' in config and config['name']:
        config['name'] = f"{config['name']}_{timestamp}"
        logger.info(f"配置文件实验名称，添加时间戳: {config['name']}")
    else:
        config['name'] = f"predict_{timestamp}"
        logger.info(f"设置默认实验名称: {config['name']}")
    
    # 预测参数
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
        logger.info(f"命令行覆盖图像尺寸: {args.imgsz}")
    
    if args.conf is not None:
        config['conf'] = args.conf
        logger.info(f"命令行覆盖置信度阈值: {args.conf}")
    
    if args.iou is not None:
        config['iou'] = args.iou
        logger.info(f"命令行覆盖IoU阈值: {args.iou}")
    
    if args.max_det is not None:
        config['max_det'] = args.max_det
        logger.info(f"命令行覆盖最大检测数: {args.max_det}")
    
    if args.device is not None:
        config['device'] = args.device
        logger.info(f"命令行覆盖设备: {args.device}")
    
    if args.half:
        config['half'] = True
        logger.info("启用FP16半精度推理")
    
    # 输出配置
    if args.save:
        config['save'] = True
        logger.info("启用结果保存")
    
    if args.save_json:
        config['save_json'] = True
        logger.info("启用JSON保存")
    
    if args.save_txt:
        config['save_txt'] = True
        logger.info("启用YOLO格式标注文件保存")
    
    if args.project is not None:
        config['project'] = args.project
        logger.info(f"命令行覆盖项目目录: {args.project}")
    
    if args.exist_ok:
        config['exist_ok'] = True
        logger.info("允许覆盖已存在的实验目录")
    
    # 显示配置
    if args.show:
        config['show'] = True
        logger.info("启用结果显示")
    
    if args.verbose:
        config['verbose'] = True
        logger.info("启用详细日志")
    
    return config


def get_source_info(source_path):
    """获取输入源信息"""
    source_path = Path(source_path)
    
    if source_path.is_file():
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            return {'type': 'image', 'path': str(source_path)}
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            return {'type': 'video', 'path': str(source_path)}
        else:
            return {'type': 'file', 'path': str(source_path)}
    elif source_path.is_dir():
        return {'type': 'directory', 'path': str(source_path)}
    elif str(source_path).startswith(('http://', 'https://')):
        return {'type': 'url', 'path': str(source_path)}
    elif str(source_path).isdigit():
        return {'type': 'webcam', 'path': str(source_path)}
    else:
        return {'type': 'unknown', 'path': str(source_path)}


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
    
    if 'source' not in config or not config['source']:
        logger.error("错误：必须指定输入源（通过 --source 参数或配置文件）")
        sys.exit(1)
    
    # 加载模型
    logger.info(f"加载模型: {config['model']}")
    model = YOLO(config['model'])
    
    # 确定任务类型
    task_type = model.task
    logger.info(f"模型任务类型: {task_type}")
    
    # 获取输入源信息
    source_info = get_source_info(config['source'])
    logger.info(f"输入源类型: {source_info['type']}")
    
    # 准备预测参数
    predict_args = {}
    excluded_keys = ['model', 'source', 'save_json', 'save_txt']
    
    for key, value in config.items():
        if key in excluded_keys:
            continue
        predict_args[key] = value
    
    # 创建保存目录
    save_dir = None
    labels_dir = None
    if config.get('save_json', False) or config.get('save_txt', False):
        project = config.get('project', 'runs/predict')
        name = config.get('name', 'predict')
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if config.get('save_json', False):
            logger.info(f"JSON文件将保存到: {save_dir}")
        
        if config.get('save_txt', False):
            labels_dir = save_dir / 'labels'
            labels_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"YOLO标注文件将保存到: {labels_dir}")
    
    try:
        logger.info("开始预测...")
        logger.info(f"模型: {config['model']}")
        logger.info(f"输入源: {config['source']}")
        
        # 执行预测
        results = model.predict(source=config['source'], **predict_args)
        
        # 处理结果
        processed_count = 0
        json_saved_count = 0
        txt_saved_count = 0
        total_detections = 0
        detection_summary = {}
        
        for result in results:
            processed_count += 1
            
            # 统计检测结果
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                total_detections += num_detections
                
                # 统计每个类别的检测数量
                if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                    for cls_id in result.boxes.cls:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1
            
            # 保存JSON文件
            if config.get('save_json', False) and save_dir:
                if process_single_result(result, save_dir, processed_count):
                    json_saved_count += 1
            
            # 保存YOLO格式标注文件
            if config.get('save_txt', False) and labels_dir:
                if process_single_result_yolo(result, labels_dir, processed_count):
                    txt_saved_count += 1
            
            # 定期清理内存
            if processed_count % 50 == 0:
                cleanup_memory()
                logger.info(f"已处理 {processed_count} 张图片")
        
        # 打印预测结果摘要
        logger.info("=== 预测结果摘要 ===")
        logger.info(f"处理图像数: {processed_count}")
        logger.info(f"总检测数: {total_detections}")
        
        if detection_summary:
            logger.info("类别统计:")
            for cls_name, count in sorted(detection_summary.items()):
                logger.info(f"  {cls_name}: {count}")
        
        if config.get('save_json', False):
            logger.info(f"成功保存 {json_saved_count} 个JSON文件")
        
        if config.get('save_txt', False):
            logger.info(f"成功保存 {txt_saved_count} 个YOLO格式标注文件")
        
        # 最终清理
        cleanup_memory()
        logger.info("预测完成！")
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        import traceback
        logger.error(f"错误堆栈跟踪:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
