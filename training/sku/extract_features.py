#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKU 特征提取脚本
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from oml.models import ViTExtractor
except ImportError:
    raise ImportError("请先安装 OML: pip install open-metric-learning")


def main():
    parser = argparse.ArgumentParser(description="SKU 特征提取")
    parser.add_argument("--input", "-i", default="d:/A_pack/pack/data/sku_library", help="sku_library 目录路径")
    parser.add_argument("--csv", "-c", default="sku_library.csv", help="指定 CSV 文件")
    parser.add_argument("--output", "-o", default=None, help="输出目录，默认与 input 相同")
    parser.add_argument("--weights", "-w", default="vits16_dino",
                        help="模型权重标识或本地 .pth/.pt 路径")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="推理 batch size")
    parser.add_argument("--device", default="cpu", help="cuda / cpu / auto")
    parser.add_argument("--no-l2", action="store_true", help="禁用 L2 归一化")
    args = parser.parse_args()

    input_dir = Path(args.input)
    csv_path = input_dir / args.csv  # ← 用 args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}")

    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"加载 {len(df)} 张图片，共 {df['sku_id'].nunique()} 个 SKU")

    # 路径转绝对路径（关键！）
    df["path"] = df["path"].apply(lambda p: str(input_dir / p))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    print(f"使用设备: {device}")

    print(f"加载模型: {args.weights}")
    if args.weights.endswith(".pth") or args.weights.endswith(".pt"):
        model = ViTExtractor.from_pretrained("vits16_dino")
        state_dict = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = ViTExtractor.from_pretrained(args.weights)
    model = model.to(device).eval()

    features = []
    batch_tensors = []

    def flush_batch():
        nonlocal batch_tensors, features
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            embs = model(batch)
        embs = embs.cpu().numpy()
        if not args.no_l2:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / (norms + 1e-8)
        features.extend(embs)
        batch_tensors.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        img_path = Path(row["path"])  # ← 已经是绝对路径
        if not img_path.exists():
            print(f"[WARN] 图片不存在: {img_path}，补零向量")
            features.append(np.zeros(384, dtype=np.float32))
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        batch_tensors.append(tensor)

        if len(batch_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

    features = np.stack(features, axis=0).astype(np.float32)
    assert features.shape[0] == len(df)

    npy_path = output_dir / "sku_features.npy"
    np.save(npy_path, features)

    meta = {
        "shape": list(features.shape),
        "csv_rows": len(df),
        "l2_norm": not args.no_l2,
        "model": args.weights,
        "mean_norm": float(np.mean(np.linalg.norm(features, axis=1)))
    }
    with open(output_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！")
    print(f"   特征矩阵: {features.shape}")
    print(f"   保存路径: {npy_path}")


if __name__ == "__main__":
    main()