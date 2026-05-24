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
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pack_compare.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

BEIJING_TZ = datetime.now().astimezone().tzinfo

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_yolo_labels(label_path, image_width, image_height):
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            if len(coords) == 4:
                x_center, y_center, w, h = coords
                x1 = (x_center - w / 2) * image_width
                y1 = (y_center - h / 2) * image_height
                x2 = (x_center + w / 2) * image_width
                y2 = (y_center + h / 2) * image_height
                labels.append({
                    'class_id': cls,
                    'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'type': 'bbox'
                })
            else:
                label = {'class_id': cls, 'type': 'segmentation', 'segments': coords}
                labels.append(label)
    
    return labels

def extract_detections(result):
    detections = []
    
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            detection = {}
            
            if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                detection['confidence'] = result.boxes.conf[i].item()
            
            if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                detection['class_id'] = int(result.boxes.cls[i].item())
            
            if hasattr(result, 'names') and result.names:
                detection['name'] = result.names.get(detection.get('class_id', 0), 'object')
            
            if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                box = result.boxes.xyxy[i].cpu().numpy()
                detection['box'] = {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            
            detections.append(detection)
    
    return detections

def extract_detections_with_masks(result):
    detections = []
    
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            detection = {}
            
            if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                detection['confidence'] = result.boxes.conf[i].item()
            
            if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                detection['class_id'] = int(result.boxes.cls[i].item())
            
            if hasattr(result, 'names') and result.names:
                detection['name'] = result.names.get(detection.get('class_id', 0), 'object')
            
            if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                box = result.boxes.xyxy[i].cpu().numpy()
                detection['box'] = {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            
            if hasattr(result, 'masks') and result.masks is not None:
                if i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    detection['mask'] = mask
            
            detections.append(detection)
    
    return detections

def fallback_visualize(image_path, detections, model_name, save_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return
        
        color = (0, 0, 255) if model_name == 'model1' else (0, 255, 0)
        text_color = (255, 255, 255)
        
        for det in detections:
            if 'box' in det:
                box = det['box']
                x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det.get('name', 'object')} {det.get('confidence', 0):.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_width, label_height = label_size
                
                cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        cv2.imwrite(save_path, image)
    except Exception as e:
        logger.error(f"备用可视化失败: {e}")

def calculate_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def visualize_detections(image_path, detections, model_name, save_path, class_names=None):
    try:
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("simhei.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            color = (255, 0, 0) if model_name == 'model1' else (0, 255, 0)
            
            if 'mask' in det and det['mask'] is not None:
                mask = det['mask']
                mask = cv2.resize(mask, (image.width, image.height))
                mask_image = Image.fromarray((mask * 128).astype(np.uint8))
                colored_mask = Image.new('RGBA', image.size, (color[0], color[1], color[2], 80))
                mask = mask > 0.5
                for y in range(image.height):
                    for x in range(image.width):
                        if mask[y, x]:
                            image.putpixel((x, y), color)
            
            if 'box' in det:
                box = det['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                cls_name = det.get('name', 'object')
                if class_names and 'class_id' in det:
                    cls_name = class_names.get(det['class_id'], cls_name)
                label = f"{cls_name}: {det.get('confidence', 0):.2f}"
                
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)
                draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        
        image.save(save_path)
        return True
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO模型效果对比脚本（支持分割任务）', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model1', type=str, required=True, help='第一个模型权重文件路径')
    parser.add_argument('--model2', type=str, required=True, help='第二个模型权重文件路径')
    parser.add_argument('--source', type=str, required=True, help='测试图片目录路径')
    parser.add_argument('--labels', type=str, help='YOLO格式标注目录路径（可选）')
    parser.add_argument('--dataset-yaml', type=str, help='数据集配置文件路径（包含类别信息）')
    parser.add_argument('--output', type=str, default='runs/compare', help='输出目录路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸（与训练保持一致）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--diff-threshold', type=float, default=0.2, help='差异显著阈值')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备（cpu或GPU编号）')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理（CPU不支持）')
    parser.add_argument('--batch', type=int, default=16, help='批次大小（与训练保持一致）')
    parser.add_argument('--focus-class', type=int, help='关注的类别ID（如3表示Carton-outer-occlusion）')
    parser.add_argument('--enable-val', action='store_true', help='启用YOLO内置评估（CPU环境建议关闭以提高速度）')
    
    return parser.parse_args()

def generate_html_report(results, output_dir, diff_threshold, model1_metrics, model2_metrics, class_names):
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO模型效果对比报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .summary table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        .summary th, .summary td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        .summary th {{ background-color: #4CAF50; color: white; }}
        .metrics-compare {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
        .metrics-card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 250px; }}
        .metrics-card.model1 {{ border-top: 4px solid #f44336; }}
        .metrics-card.model2 {{ border-top: 4px solid #4CAF50; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 20px; }}
        .image-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .image-card.different {{ border: 3px solid #f44336; }}
        .image-row {{ display: flex; gap: 10px; margin-bottom: 10px; }}
        .image-item {{ flex: 1; }}
        .image-item img {{ width: 100%; border-radius: 4px; }}
        .image-item p {{ text-align: center; margin: 5px 0; font-weight: bold; }}
        .metrics {{ background: #f9f9f9; padding: 10px; border-radius: 4px; }}
        .metrics p {{ margin: 5px 0; font-size: 14px; }}
        .diff-highlight {{ color: #f44336; font-weight: bold; }}
        .normal {{ color: #4CAF50; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; }}
        .better {{ color: #2196F3; font-weight: bold; }}
        .class-info {{ background: #e8f5e9; padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLO模型效果对比报告</h1>
        <p>生成时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="class-info">
            <strong>类别信息:</strong> {', '.join([f'{k}: {v}' for k, v in class_names.items()])}
        </div>
        
        <h2>📊 整体指标对比（YOLO内置评估）</h2>
        <div class="metrics-compare">
            <div class="metrics-card model1">
                <h3>模型1</h3>
                <p><strong>mAP@0.5:</strong> {model1_metrics.get('mAP50', 'N/A')}</p>
                <p><strong>mAP@0.5:0.95:</strong> {model1_metrics.get('mAP50_95', 'N/A')}</p>
                <p><strong>Precision:</strong> {model1_metrics.get('precision', 'N/A')}</p>
                <p><strong>Recall:</strong> {model1_metrics.get('recall', 'N/A')}</p>
                {model1_metrics.get('mask_mAP50', '') and f"<p><strong>Mask mAP@0.5:</strong> {model1_metrics['mask_mAP50']}</p>" or ''}
            </div>
            <div class="metrics-card model2">
                <h3>模型2</h3>
                <p><strong>mAP@0.5:</strong> {model2_metrics.get('mAP50', 'N/A')}</p>
                <p><strong>mAP@0.5:0.95:</strong> {model2_metrics.get('mAP50_95', 'N/A')}</p>
                <p><strong>Precision:</strong> {model2_metrics.get('precision', 'N/A')}</p>
                <p><strong>Recall:</strong> {model2_metrics.get('recall', 'N/A')}</p>
                {model2_metrics.get('mask_mAP50', '') and f"<p><strong>Mask mAP@0.5:</strong> {model2_metrics['mask_mAP50']}</p>" or ''}
            </div>
        </div>
        
        <h2>📈 统计摘要</h2>
        <table>
            <tr>
                <th>统计项</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>测试图片总数</td>
                <td>{len(results)}</td>
            </tr>
            <tr>
                <td>差异显著图片数</td>
                <td>{sum(1 for r in results if r['is_significant_diff'])}</td>
            </tr>
            <tr>
                <td>模型1平均检测数</td>
                <td>{np.mean([r['model1_detections'] for r in results]):.2f}</td>
            </tr>
            <tr>
                <td>模型2平均检测数</td>
                <td>{np.mean([r['model2_detections'] for r in results]):.2f}</td>
            </tr>
        </table>
    </div>
    
    <h2>🖼️ 图片对比结果</h2>
    <div class="image-grid">
"""
    
    for result in results:
        is_diff = result['is_significant_diff']
        diff_class = 'different' if is_diff else ''
        highlight_text = '<span class="diff-highlight">⚠️ 差异显著</span>' if is_diff else '<span class="normal">✅ 效果一致</span>'
        
        html_content += f"""
        <div class="image-card {diff_class}">
            <h3>{result['image_name']} {highlight_text}</h3>
            <div class="image-row">
                <div class="image-item">
                    <p>模型1 <span style="color: #f44336;">(红框/红色mask)</span></p>
                    <img src="{result['model1_visual_path']}" alt="模型1检测结果">
                    <p>检测数: {result['model1_detections']}</p>
                </div>
                <div class="image-item">
                    <p>模型2 <span style="color: #4CAF50;">(绿框/绿色mask)</span></p>
                    <img src="{result['model2_visual_path']}" alt="模型2检测结果">
                    <p>检测数: {result['model2_detections']}</p>
                </div>
            </div>
            <div class="metrics">
                <p><strong>检测数差异:</strong> {abs(result['model1_detections'] - result['model2_detections'])}</p>
                <p><strong>置信度差异:</strong> {result['confidence_diff']:.4f}</p>
                {result.get('mask_iou_diff', '') and f"<p><strong>Mask IoU差异:</strong> {result['mask_iou_diff']:.4f}</p>" or ''}
            </div>
        </div>
        """
    
    html_content += """
    </div>
    
    <div class="footer">
        <p>YOLO模型对比报告 - 差异阈值: {} | 置信度阈值: {}</p>
        <p>指标计算方式: YOLO内置评估机制（与train/val脚本一致）</p>
    </div>
</body>
</html>
    """.format(diff_threshold, args.conf)
    
    report_path = output_dir / 'compare_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML报告已生成: {report_path}")

def main():
    global args
    args = parse_arguments()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visual_dir = output_dir / 'visualizations'
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = output_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    model1_labels_dir = labels_dir / 'model1'
    model1_labels_dir.mkdir(parents=True, exist_ok=True)
    
    model2_labels_dir = labels_dir / 'model2'
    model2_labels_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = {}
    if args.dataset_yaml and Path(args.dataset_yaml).exists():
        with open(args.dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
            if 'names' in dataset_config:
                names = dataset_config['names']
                if isinstance(names, list):
                    class_names = {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
        logger.info(f"加载类别信息: {class_names}")
    
    logger.info(f"加载模型1: {args.model1}")
    model1 = YOLO(args.model1)
    
    logger.info(f"加载模型2: {args.model2}")
    model2 = YOLO(args.model2)
    
    source_path = Path(args.source)
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_files.extend(sorted(source_path.glob(ext)))
    
    if not image_files:
        logger.error(f"在 {args.source} 中未找到图片文件")
        sys.exit(1)
    
    logger.info(f"找到 {len(image_files)} 张测试图片")
    
    label_path = Path(args.labels) if args.labels else None
    if label_path and not label_path.exists():
        logger.warning(f"标注目录不存在: {args.labels}")
        label_path = None
    
    if args.dataset_yaml and Path(args.dataset_yaml).exists():
        temp_yaml_path = args.dataset_yaml
        logger.info(f"使用数据集配置文件: {temp_yaml_path}")
    else:
        temp_yaml_path = None
        logger.warning("未提供数据集配置文件，将跳过整体指标评估")
    
    model1_metrics = {}
    model2_metrics = {}
    
    if args.enable_val and temp_yaml_path:
        logger.info("注意：YOLO内置评估在CPU环境下可能非常慢，请耐心等待...")
        try:
            val_args = {
                'data': temp_yaml_path,
                'imgsz': args.imgsz,
                'conf': args.conf,
                'iou': args.iou,
                'device': args.device,
                'half': False if args.device == 'cpu' else args.half,
                'batch': args.batch,
                'save_json': False,
                'plots': False,
                'verbose': False
            }
            
            logger.info("使用YOLO内置评估机制评估模型1...")
            result1_val = model1.val(**val_args)
            if hasattr(result1_val, 'results_dict'):
                model1_metrics = {
                    'mAP50': f"{result1_val.results_dict.get('metrics/mAP50(B)', 0):.4f}",
                    'mAP50_95': f"{result1_val.results_dict.get('metrics/mAP50-95(B)', 0):.4f}",
                    'precision': f"{result1_val.results_dict.get('metrics/precision(B)', 0):.4f}",
                    'recall': f"{result1_val.results_dict.get('metrics/recall(B)', 0):.4f}"
                }
                if 'metrics/mAP50(M)' in result1_val.results_dict:
                    model1_metrics['mask_mAP50'] = f"{result1_val.results_dict['metrics/mAP50(M)']:.4f}"
            logger.info(f"模型1评估完成 - mAP@0.5: {model1_metrics.get('mAP50', 'N/A')}")
            
            logger.info("使用YOLO内置评估机制评估模型2...")
            result2_val = model2.val(**val_args)
            if hasattr(result2_val, 'results_dict'):
                model2_metrics = {
                    'mAP50': f"{result2_val.results_dict.get('metrics/mAP50(B)', 0):.4f}",
                    'mAP50_95': f"{result2_val.results_dict.get('metrics/mAP50-95(B)', 0):.4f}",
                    'precision': f"{result2_val.results_dict.get('metrics/precision(B)', 0):.4f}",
                    'recall': f"{result2_val.results_dict.get('metrics/recall(B)', 0):.4f}"
                }
                if 'metrics/mAP50(M)' in result2_val.results_dict:
                    model2_metrics['mask_mAP50'] = f"{result2_val.results_dict['metrics/mAP50(M)']:.4f}"
            logger.info(f"模型2评估完成 - mAP@0.5: {model2_metrics.get('mAP50', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"YOLO内置评估失败，将跳过整体指标计算: {e}")
    else:
        if not args.enable_val:
            logger.info("未启用YOLO内置评估(--enable-val)，将跳过整体指标计算...")
        if not temp_yaml_path:
            logger.info("未提供数据集配置文件，将跳过整体指标评估")
    
    cleanup_memory()
    
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"处理第 {idx}/{len(image_files)} 张图片: {image_path.name}")
        
        try:
            predict_args = {
                'imgsz': args.imgsz,
                'conf': args.conf,
                'iou': args.iou,
                'device': args.device,
                'half': False if args.device == 'cpu' else args.half,
                'show': False,
                'save': False,
                'stream': False,
                'verbose': False
            }
            
            result1 = model1.predict(source=str(image_path), **predict_args)[0]
            result2 = model2.predict(source=str(image_path), **predict_args)[0]
            
            detections1_for_json = extract_detections(result1)
            detections2_for_json = extract_detections(result2)
            
            if args.focus_class is not None:
                detections1_for_json = [d for d in detections1_for_json if d.get('class_id') == args.focus_class]
                detections2_for_json = [d for d in detections2_for_json if d.get('class_id') == args.focus_class]
            
            if len(detections1_for_json) > 0 and len(detections2_for_json) > 0:
                avg_conf1 = sum(d['confidence'] for d in detections1_for_json) / len(detections1_for_json)
                avg_conf2 = sum(d['confidence'] for d in detections2_for_json) / len(detections2_for_json)
                confidence_diff = abs(avg_conf1 - avg_conf2)
            else:
                confidence_diff = 1.0 if len(detections1_for_json) != len(detections2_for_json) else 0.0
            
            mask_iou_diff = 0.0
            if hasattr(result1, 'masks') and result1.masks is not None and len(result1.masks) > 0:
                if hasattr(result2, 'masks') and result2.masks is not None and len(result2.masks) > 0:
                    mask1 = result1.masks.data[0].cpu().numpy()
                    mask2 = result2.masks.data[0].cpu().numpy()
                    mask1 = cv2.resize(mask1, (mask2.shape[2], mask2.shape[1]))
                    mask_iou_diff = 1.0 - calculate_mask_iou(mask1, mask2)
            
            detection_count_diff = abs(len(detections1_for_json) - len(detections2_for_json)) / max(len(detections1_for_json), len(detections2_for_json), 1)
            is_significant_diff = confidence_diff >= args.diff_threshold or detection_count_diff >= args.diff_threshold or mask_iou_diff >= args.diff_threshold
            
            model1_visual_path = visual_dir / f"{image_path.stem}_model1.jpg"
            model2_visual_path = visual_dir / f"{image_path.stem}_model2.jpg"
            
            try:
                plotted_img1 = result1.plot()
                if plotted_img1 is not None and len(plotted_img1.shape) == 3:
                    cv2.imwrite(str(model1_visual_path), plotted_img1)
            except Exception as e:
                logger.warning(f"模型1可视化失败: {e}")
                fallback_visualize(str(image_path), detections1_for_json, 'model1', str(model1_visual_path))
            
            try:
                plotted_img2 = result2.plot()
                if plotted_img2 is not None and len(plotted_img2.shape) == 3:
                    cv2.imwrite(str(model2_visual_path), plotted_img2)
            except Exception as e:
                logger.warning(f"模型2可视化失败: {e}")
                fallback_visualize(str(image_path), detections2_for_json, 'model2', str(model2_visual_path))
            
            model1_label_path = model1_labels_dir / f"{image_path.stem}.json"
            model2_label_path = model2_labels_dir / f"{image_path.stem}.json"
            
            with open(model1_label_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': str(image_path),
                    'detections': detections1_for_json,
                    'total_detections': len(detections1_for_json)
                }, f, indent=2, ensure_ascii=False)
            
            with open(model2_label_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': str(image_path),
                    'detections': detections2_for_json,
                    'total_detections': len(detections2_for_json)
                }, f, indent=2, ensure_ascii=False)
            
            result_entry = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'model1_detections': len(detections1_for_json),
                'model2_detections': len(detections2_for_json),
                'model1_visual_path': str(model1_visual_path.relative_to(output_dir)),
                'model2_visual_path': str(model2_visual_path.relative_to(output_dir)),
                'confidence_diff': confidence_diff,
                'is_significant_diff': is_significant_diff
            }
            if mask_iou_diff > 0:
                result_entry['mask_iou_diff'] = mask_iou_diff
            
            all_results.append(result_entry)
            
            if idx % 10 == 0:
                cleanup_memory()
        
        except Exception as e:
            logger.error(f"处理图片 {image_path.name} 时出错: {e}")
            continue
    
    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    significant_cases = [r for r in all_results if r['is_significant_diff']]
    with open(output_dir / 'significant_diff_cases.json', 'w', encoding='utf-8') as f:
        json.dump(significant_cases, f, indent=2, ensure_ascii=False)
    
    logger.info(f"共处理 {len(all_results)} 张图片")
    logger.info(f"发现 {len(significant_cases)} 张差异显著的图片")
    
    generate_html_report(all_results, output_dir, args.diff_threshold, model1_metrics, model2_metrics, class_names)
    
    cleanup_memory()
    logger.info("模型对比完成！")

if __name__ == '__main__':
    main()