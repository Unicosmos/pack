
"""
切分 sku_library.csv 为 train.csv 和 val.csv

使用方法：
    python split_train_val.py --input ../sku_library/sku_library.csv --output_dir ../sku_library/ --val_ratio 0.2
"""

import argparse
import pandas as pd
from pathlib import Path
import random

def main():
    parser = argparse.ArgumentParser(description="切分数据集")
    parser.add_argument("--input", "-i",type=str, default="../sku_library/sku_library.csv", help="输入CSV文件路径")
    parser.add_argument("--output_dir","-o", type=str, default="../sku_library/", help="输出目录")
    parser.add_argument("--val_ratio","-v",type=float, default=0.2, help="验证集比例")
    args = parser.parse_args()
    
    random.seed(42)
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_path)
    print(f"加载 {len(df)} 张图片")
    
    # 按 label 来切分，不是按图片！
    labels = list(df["label"].unique())
    print(f"发现 {len(labels)} 个label")
    
    # 打乱label列表
    random.shuffle(labels)
    
    # 计算验证集的label数量
    num_val_labels = max(1, int(len(labels) * args.val_ratio))
    val_labels = labels[:num_val_labels]
    train_labels = labels[num_val_labels:]
    
    # 切分数据集
    train_df = df[df["label"].isin(train_labels)].reset_index(drop=True)
    val_df = df[df["label"].isin(val_labels)].reset_index(drop=True)
    
    # 保存
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n训练集: {len(train_df)} 张")
    print(f"验证集: {len(val_df)} 张")
    print(f"\ntrain.csv -> {train_path}")
    print(f"val.csv -> {val_path}")

if __name__ == "__main__":
    main()
