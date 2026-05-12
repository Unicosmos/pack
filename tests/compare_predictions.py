import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_yolo_label(label_path):
    """解析YOLO格式的标注文件"""
    detections = []
    if not label_path.exists():
        return detections
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                detections.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return detections


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2
    
    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_detections(detections1, detections2, iou_threshold=0.5):
    """匹配两个检测结果列表"""
    matches = []
    unmatched1 = list(range(len(detections1)))
    unmatched2 = list(range(len(detections2)))
    
    for i in range(len(detections1)):
        best_iou = 0
        best_j = -1
        for j in range(len(detections2)):
            if j not in unmatched2:
                continue
            if detections1[i]['class_id'] != detections2[j]['class_id']:
                continue
            iou = calculate_iou(detections1[i], detections2[j])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_j = j
        if best_j != -1:
            matches.append((i, best_j, best_iou))
            unmatched1.remove(i)
            unmatched2.remove(best_j)
    
    return matches, unmatched1, unmatched2


def compare_predictions(label_dir1, label_dir2, output_dir, iou_threshold=0.5):
    """比较两个模型的预测结果"""
    dir1 = Path(label_dir1)
    dir2 = Path(label_dir2)
    
    if not dir1.exists():
        print(f"错误：目录不存在: {label_dir1}")
        return
    
    if not dir2.exists():
        print(f"错误：目录不存在: {label_dir2}")
        return
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    txt_files1 = set(f for f in dir1.iterdir() if f.suffix == '.txt')
    txt_files2 = set(f for f in dir2.iterdir() if f.suffix == '.txt')
    
    all_files = set(f.stem for f in txt_files1) | set(f.stem for f in txt_files2)
    
    diff_summary = []
    detailed_diff = []
    
    for file_stem in all_files:
        file1 = dir1 / f"{file_stem}.txt"
        file2 = dir2 / f"{file_stem}.txt"
        
        det1 = parse_yolo_label(file1)
        det2 = parse_yolo_label(file2)
        
        matches, unmatched1, unmatched2 = match_detections(det1, det2, iou_threshold)
        
        diff_info = {
            'filename': f"{file_stem}.jpg",
            'det1_count': len(det1),
            'det2_count': len(det2),
            'matched_count': len(matches),
            'unmatched_in_model1': len(unmatched1),
            'unmatched_in_model2': len(unmatched2),
            'unmatched1_indices': unmatched1,
            'unmatched2_indices': unmatched2,
            'matches': matches
        }
        
        has_diff = len(unmatched1) > 0 or len(unmatched2) > 0
        if has_diff:
            detailed_diff.append(diff_info)
        
        diff_summary.append(diff_info)
    
    stats = {
        'total_files': len(all_files),
        'files_with_diff': len(detailed_diff),
        'avg_det1_count': sum(d['det1_count'] for d in diff_summary) / len(diff_summary) if diff_summary else 0,
        'avg_det2_count': sum(d['det2_count'] for d in diff_summary) / len(diff_summary) if diff_summary else 0,
        'total_unmatched1': sum(d['unmatched_in_model1'] for d in diff_summary),
        'total_unmatched2': sum(d['unmatched_in_model2'] for d in diff_summary)
    }
    
    with open(output / 'diff_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型预测差异比较报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("统计摘要:\n")
        f.write(f"  总文件数: {stats['total_files']}\n")
        f.write(f"  有差异的文件数: {stats['files_with_diff']}\n")
        f.write(f"  Model 1 平均检测数: {stats['avg_det1_count']:.2f}\n")
        f.write(f"  Model 2 平均检测数: {stats['avg_det2_count']:.2f}\n")
        f.write(f"  Model 1 独有检测数: {stats['total_unmatched1']}\n")
        f.write(f"  Model 2 独有检测数: {stats['total_unmatched2']}\n")
        f.write("\n")
        
        f.write("详细差异列表:\n")
        f.write("-" * 60 + "\n")
        
        for diff in detailed_diff:
            f.write(f"\n文件: {diff['filename']}\n")
            f.write(f"  Model 1 检测数: {diff['det1_count']}\n")
            f.write(f"  Model 2 检测数: {diff['det2_count']}\n")
            f.write(f"  匹配数: {diff['matched_count']}\n")
            
            if diff['unmatched_in_model1'] > 0:
                f.write(f"  Model 1 独有检测: {diff['unmatched_in_model1']} 个\n")
            
            if diff['unmatched_in_model2'] > 0:
                f.write(f"  Model 2 独有检测: {diff['unmatched_in_model2']} 个\n")
    
    with open(output / 'diff_files.txt', 'w', encoding='utf-8') as f:
        for diff in detailed_diff:
            f.write(f"{diff['filename']}\n")
    
    print(f"比较完成！结果已保存到 {output}")
    print(f"总文件数: {stats['total_files']}")
    print(f"有差异的文件数: {stats['files_with_diff']}")
    print(f"Model 1 独有检测数: {stats['total_unmatched1']}")
    print(f"Model 2 独有检测数: {stats['total_unmatched2']}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='比较两个模型在同一数据集上的预测差异')
    parser.add_argument('label_dir1', type=str, help='第一个模型的labels目录路径')
    parser.add_argument('label_dir2', type=str, help='第二个模型的labels目录路径')
    parser.add_argument('--output', type=str, default='diff_result', help='输出目录')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU匹配阈值')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    compare_predictions(args.label_dir1, args.label_dir2, args.output, args.iou_threshold) 

# python compare_predictions.py  d:\A_pack\pack\YOLO\runs\predict\lscd_predict_20260512191204\labels  d:\A_pack\pack\YOLO\runs\predict\lscd_predict_20260512191332\labels --output diff_result --iou-threshold 0.5
