#!/usr/bin/env python3
"""
查找被增强的样本
比较增强前后的标注文件内容，找出被修改的样本
"""
import os
import sys
from pathlib import Path


def main():
    """主函数"""
    # 路径设置
    original_labels_dir = Path('/root/source/data2/hyg/projects/pack/coco_style_fourclass/yolo_dataset_seg/labels/train')
    augmented_labels_dir = Path('/root/source/data2/hyg/projects/pack/coco_style_fourclass/augmented/labels/train')
    
    # 检查目录是否存在
    if not original_labels_dir.exists():
        print(f"原始标注目录不存在: {original_labels_dir}")
        return
    
    if not augmented_labels_dir.exists():
        print(f"增强标注目录不存在: {augmented_labels_dir}")
        return
    
    # 收集所有标注文件
    original_files = set(f.name for f in original_labels_dir.glob('*.txt'))
    augmented_files = set(f.name for f in augmented_labels_dir.glob('*.txt'))
    
    # 找出共同的文件
    common_files = original_files.intersection(augmented_files)
    print(f"找到 {len(common_files)} 个共同的标注文件")
    
    # 比较文件内容
    augmented_samples = []
    
    for filename in common_files:
        original_path = original_labels_dir / filename
        augmented_path = augmented_labels_dir / filename
        
        # 读取文件内容
        with open(original_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(augmented_path, 'r', encoding='utf-8') as f:
            augmented_content = f.read()
        
        # 比较内容
        if original_content != augmented_content:
            augmented_samples.append(filename)
    
    # 在输出结果前添加排序
    augmented_samples.sort()

    # 输出结果
    print(f"\n找到 {len(augmented_samples)} 个被增强的样本:")
    
    # 输出前20个样本作为示例
    print("\n前20个被增强的样本:")
    for i, sample in enumerate(augmented_samples[:20], 1):
        print(f"{i:3d}. {sample}")
    
    # 保存结果到文件
    output_file = Path('augmented_samples.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('被增强的样本列表:\n')
        f.write('=' * 50 + '\n')
        for sample in augmented_samples:
            f.write(sample + '\n')
    
    print(f"\n完整列表已保存到: {output_file}")
    print(f"总共有 {len(augmented_samples)} 个样本被增强")


if __name__ == '__main__':
    main()
