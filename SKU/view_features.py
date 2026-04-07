"""
查看特征提取结果

使用方法:
    python view_features.py --input features.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_features(path: str):
    """加载特征文件"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_feature_info(results):
    """打印特征信息"""
    print("=" * 70)
    print("特征提取结果详情")
    print("=" * 70)
    
    total_items = len(results)
    print(f"\n总条目数: {total_items}")
    
    if total_items == 0:
        print("文件为空")
        return
    
    print("\n" + "-" * 70)
    print("每条记录结构:")
    print("-" * 70)
    
    sample = results[0]
    print(f"键: {list(sample.keys())}")
    
    if 'feature_vector' in sample:
        fv = sample['feature_vector']
        if fv is not None:
            print(f"  feature_vector 类型: {type(fv)}")
            print(f"  feature_vector 形状: {fv.shape}")
            print(f"  feature_vector 前10个值: {fv[:10]}")
    
    if 'detections' in sample:
        print(f"  detections 数量: {len(sample['detections'])}")
        if len(sample['detections']) > 0:
            det = sample['detections'][0]
            print(f"  detection 键: {list(det.keys())}")
            if 'feature_vector' in det and det['feature_vector'] is not None:
                fv = det['feature_vector']
                print(f"    feature_vector 形状: {fv.shape}")
                print(f"    feature_vector 前10个值: {fv[:10]}")
    
    print("\n" + "-" * 70)
    print("所有条目概览:")
    print("-" * 70)
    
    feature_dims = []
    feature_means = []
    feature_stds = []
    
    for item in results:
        if 'feature_vector' in item and item['feature_vector'] is not None:
            fv = item['feature_vector']
            feature_dims.append(fv.shape[0])
            feature_means.append(np.mean(fv))
            feature_stds.append(np.std(fv))
        
        if 'detections' in item:
            for det in item['detections']:
                if 'feature_vector' in det and det['feature_vector'] is not None:
                    fv = det['feature_vector']
                    feature_dims.append(fv.shape[0])
                    feature_means.append(np.mean(fv))
                    feature_stds.append(np.std(fv))
    
    if feature_dims:
        print(f"特征维度: {set(feature_dims)}")
        print(f"特征均值范围: [{min(feature_means):.4f}, {max(feature_means):.4f}]")
        print(f"特征标准差范围: [{min(feature_stds):.4f}, {max(feature_stds):.4f}]")
        print(f"有效特征数量: {len(feature_dims)}")


def export_to_csv(results, output_path: str):
    """导出特征到 CSV 文件"""
    rows = []
    
    for item in results:
        if 'feature_vector' in item and item['feature_vector'] is not None:
            row = {
                'image_path': item.get('image_path', ''),
                'detection_idx': -1,
            }
            for i, val in enumerate(item['feature_vector'][:10]):
                row[f'feat_{i}'] = val
            rows.append(row)
        
        if 'detections' in item:
            for det_idx, det in enumerate(item['detections']):
                if 'feature_vector' in det and det['feature_vector'] is not None:
                    row = {
                        'image_path': item.get('image_path', ''),
                        'detection_idx': det_idx,
                        'bbox': str(det.get('bbox', '')),
                        'class': det.get('class', ''),
                        'confidence': det.get('confidence', 0),
                        'feature_dim': len(det['feature_vector']),
                    }
                    for i, val in enumerate(det['feature_vector'][:10]):
                        row[f'feat_{i}'] = val
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n特征已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='查看特征提取结果')
    parser.add_argument('--input', '-i', type=str, default='features.pkl', help='特征文件路径')
    parser.add_argument('--export', '-e', type=str, default=None, help='导出到 CSV 文件')
    args = parser.parse_args()
    
    print(f"加载特征文件: {args.input}")
    results = load_features(args.input)
    
    print_feature_info(results)
    
    if args.export:
        export_to_csv(results, args.export)


if __name__ == '__main__':
    main()
