# split_sku.py
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--output-dir","-o", required=True)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(args.csv)
np.random.seed(args.seed)

# 按 SKU 划分
unique_skus = df['sku_id'].unique()
np.random.shuffle(unique_skus)
n_train = int(len(unique_skus) * args.train_ratio)
train_skus = unique_skus[:n_train]
test_skus = unique_skus[n_train:]

df_train = df[df['sku_id'].isin(train_skus)].copy()
df_test = df[df['sku_id'].isin(test_skus)].copy()

# 保存
import os
os.makedirs(args.output_dir, exist_ok=True)
df_train.to_csv(f"{args.output_dir}/train.csv", index=False)
df_test.to_csv(f"{args.output_dir}/test.csv", index=False)

print(f"训练 SKU: {len(train_skus)}, 测试 SKU: {len(test_skus)}")
print(f"训练图: {len(df_train)}, 测试图: {len(df_test)}")