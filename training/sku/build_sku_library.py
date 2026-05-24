#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKU 差异化离线增强 + Library 生成脚本
- 按原图数量动态决定入库增强策略
- 仅入库：遮挡(occ) + 旋转(rotate)，灰色填充
- 适配 sku_database.json（含 images 文件名列表）
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_sku_database(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def aug_occlusion(img, ratio_range=(0.05, 0.15), fill=(128, 128, 128)):
    """
    细长条边缘遮挡，模拟地堆中箱体互相挤压的遮挡效果。
    - 面积 5%-15%，避免大面积糊住型号/文字
    - 形状为细长条：横条(模拟上层压住) 或 竖条(模拟侧面挤住)
    - 位置偏好四边：上下左右边缘，几乎不会出现在中心
    """
    w, h = img.size
    ratio = random.uniform(*ratio_range)
    occ_area = int(w * h * ratio)

    if random.random() < 0.5:
        # 横条：全宽，很扁，模拟上层箱子压下来的遮挡
        occ_w = w
        occ_h = max(2, int(occ_area / w))
        # 只贴在上边缘或下边缘
        y = random.choice([0, h - occ_h])
        x = 0
    else:
        # 竖条：全高，很窄，模拟侧面相邻箱子挤过来的遮挡
        occ_h = h
        occ_w = max(2, int(occ_area / h))
        # 只贴在左边缘或右边缘
        x = random.choice([0, w - occ_w])
        y = 0

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x, y, x + occ_w, y + occ_h], fill=fill)
    return img_copy


def aug_rotate(img, angle_range=(-15, 15), fill=(128, 128, 128)):
    """旋转±15°，灰色背景填充，模拟斜视/俯视"""
    angle = random.uniform(*angle_range)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill, expand=False)


def decide_aug_plan(base_count):
    """
    差异化增强策略
    - 1-2张原图：每张生成 2 张增强（遮挡 + 旋转）
    - 3-4张原图：每张生成 1 张增强（仅遮挡）
    - >=5张原图：不生成增强
    """
    if base_count <= 2:
        return [("occ", aug_occlusion), ("rotate", aug_rotate)]
    elif base_count <= 4:
        return [("occ", aug_occlusion)]
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="SKU 差异化增强 Library 生成")
    parser.add_argument("--input", "-i", required=True, help="输入目录（含 images/ 和 sku_database.json）")
    parser.add_argument("--output", "-o", required=True, help="输出目录（sku_library）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，保证可复现")
    args = parser.parse_args()

    set_seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    img_input_dir = input_dir / "images"
    json_path = input_dir / "sku_database.json"

    if not img_input_dir.exists():
        raise FileNotFoundError(f"找不到 images 目录: {img_input_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 sku_database.json: {json_path}")

    sku_db = load_sku_database(json_path)

    img_output_dir = output_dir / "images"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_base = 0
    total_aug = 0
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 按 sku_id 排序处理
    for sku_id in tqdm(sorted(sku_db.keys()), desc="Processing SKUs"):
        sku_info = sku_db[sku_id]
        sku_name = sku_info.get("name", "")
        img_names = sku_info.get("images", [])

        sku_in_dir = img_input_dir / sku_id
        if not sku_in_dir.exists():
            print(f"[WARN] {sku_id} 目录不存在，跳过")
            continue

        # 只处理 json 里列出且真实存在的图片
        existing_files = []
        for name in img_names:
            fpath = sku_in_dir / name
            if fpath.exists() and fpath.suffix.lower() in valid_ext:
                existing_files.append(fpath)
            else:
                print(f"[WARN] {sku_id}/{name} 不存在或格式不支持，跳过")

        if not existing_files:
            print(f"[WARN] {sku_id} 无有效图片，跳过")
            continue

        # 以实际存在的原图数量决定增强策略
        base_count = len(existing_files)
        aug_plan = decide_aug_plan(base_count)

        sku_out_dir = img_output_dir / sku_id
        sku_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in existing_files:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERR] 无法读取 {img_path}: {e}")
                continue

            # 统一 resize 到 224x224，适配 ViT-S16
            img = img.resize((224, 224), Image.BILINEAR)
            base_name = img_path.stem

            # 保存原图
            out_base = sku_out_dir / f"{base_name}.jpg"
            img.save(out_base, quality=95)

            records.append({
                "path": str(Path("images") / sku_id / f"{base_name}.jpg"),
                "sku_id": sku_id,
                "sku_name": sku_name,
                "label": sku_id,
                "is_base": 1,
                "parent": "",
                "aug_type": "none"
            })
            total_base += 1

            # 按策略生成增强图（仅入库增强）
            for aug_name, aug_func in aug_plan:
                try:
                    aug_img = aug_func(img)
                    # 确保尺寸（rotate 保持尺寸，但保险起见）
                    if aug_img.size != (224, 224):
                        aug_img = aug_img.resize((224, 224), Image.BILINEAR)

                    aug_filename = f"{base_name}_aug_{aug_name}.jpg"
                    aug_path = sku_out_dir / aug_filename
                    aug_img.save(aug_path, quality=95)

                    records.append({
                        "path": str(Path("images") / sku_id / aug_filename),
                        "sku_id": sku_id,
                        "sku_name": sku_name,
                        "label": sku_id,
                        "is_base": 0,
                        "parent": f"{base_name}.jpg",
                        "aug_type": aug_name
                    })
                    total_aug += 1
                except Exception as e:
                    print(f"[ERR] 增强失败 {img_path} [{aug_name}]: {e}")

    # 写入 CSV
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = output_dir / "sku_library.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 统计信息
    meta = {
        "total_images": len(df),
        "total_skus": df["sku_id"].nunique(),
        "base_images": total_base,
        "aug_images": total_aug,
        "seed": args.seed,
        "strategy": {
            "1-2_base": "occ + rotate (2 per base)",
            "3-4_base": "occ only (1 per base)",
            "5+_base": "no aug"
        }
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！输出目录: {output_dir}")
    print(f"   总图片: {meta['total_images']}（原图 {total_base}，增强 {total_aug}）")
    print(f"   SKU 数: {meta['total_skus']}")
    print(f"   CSV  : {csv_path}")


if __name__ == "__main__":
    main()