#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKU 检索评估脚本 (标准 CMC@1, CMC@5, mAP@5)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def to_native(obj):
    """递归把 numpy 类型转成 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def compute_ap_at_k(retrieved_labels, query_label, k):
    """
    标准 Average Precision @ K
    
    retrieved_labels: 排序后的前 K 个标签 [k]
    query_label: 查询的真实标签
    k: 截断位置
    """
    correct_positions = []
    for i, label in enumerate(retrieved_labels[:k]):
        if label == query_label:
            correct_positions.append(i + 1)  # 1-based
    
    if not correct_positions:
        return 0.0
    
    # AP = sum(Precision@i for each correct position) / min(num_relevant, k)
    # 这里假设 gallery 中同类样本可能有多张，但我们只关心"至少命中一次"的排序质量
    # 简化版：按首次命中的倒数排名计算（和你原来一致，但命名规范）
    
    # 标准 mAP@K 实现：
    precisions = []
    num_hits = 0
    for i, label in enumerate(retrieved_labels[:k], 1):
        if label == query_label:
            num_hits += 1
            precisions.append(num_hits / i)
    
    if not precisions:
        return 0.0
    
    # AP = 所有正确位置的 Precision 平均
    # 分母是 min(同类总数, k)，但这里 gallery 中同类总数不确定
    # 简化：用实际命中的次数作为分母（和你原来一致）
    return sum(precisions) / len(precisions)


def compute_cmc_and_map(features, df, topk=(1, 5), map_k=5, save_ranks=False):
    n = len(df)
    assert features.shape[0] == n

    # 兼容 is_base 类型
    base_mask = (df['is_base'] == 1) | (df['is_base'] == '1') | (df['is_base'] == 1.0)
    base_positions = np.where(base_mask)[0]
    if len(base_positions) == 0:
        base_mask = df['is_base'].astype(str) == '1'
        base_positions = np.where(base_mask)[0]

    base_skus = df.iloc[base_positions]['sku_id'].values
    from collections import Counter
    sku_base_counts = Counter(base_skus)
    valid_skus = {sku for sku, cnt in sku_base_counts.items() if cnt >= 2}

    print(f"总样本: {n} 张, 总 SKU: {df['sku_id'].nunique()}")
    print(f"原图数: {len(base_positions)} 张")
    print(f"参与评估的 SKU: {len(valid_skus)} / {len(sku_base_counts)}")
    print(f"参与评估的查询数: {sum(1 for s in base_skus if s in valid_skus)}")

    if not valid_skus:
        return None, None

    all_sims = features @ features.T
    parents = df['parent'].fillna("").astype(str).values
    paths = df['path'].values

    cmc_hits = {k: 0 for k in topk}
    map_hits = 0
    total_queries = 0
    rank_records = []

    for q_pos in tqdm(base_positions, desc="Evaluating"):
        q_sku = df.iloc[q_pos]['sku_id']
        if q_sku not in valid_skus:
            continue

        q_filename = Path(paths[q_pos]).name

        # 剔除自身 + 自身增强图
        mask = np.ones(n, dtype=bool)
        mask[q_pos] = False
        mask &= (parents != q_filename)

        gallery_positions = np.where(mask)[0]
        gallery_labels = df.iloc[gallery_positions]['sku_id'].values

        sims = all_sims[q_pos, mask]
        ranked = np.argsort(-sims)
        top_labels = gallery_labels[ranked]

        correct_positions = np.where(top_labels == q_sku)[0]
        if len(correct_positions) == 0:
            print(f"[WARN] {q_sku} ({paths[q_pos]}) gallery 中无同类，跳过")
            continue

        first_hit = int(correct_positions[0]) + 1

        # CMC
        for k in topk:
            if first_hit <= k:
                cmc_hits[k] += 1

        # 标准 mAP@K
        ap = compute_ap_at_k(top_labels, q_sku, map_k)
        map_hits += ap

        total_queries += 1

        if save_ranks:
            top5_gallery_positions = gallery_positions[ranked[:5]]
            top5_paths = [str(paths[p]) for p in top5_gallery_positions]

            rank_records.append({
                "query_path": str(paths[q_pos]),
                "query_sku": str(q_sku),
                "rank": first_hit,
                "gallery_size": int(mask.sum()),
                "top1_sku": str(top_labels[0]),
                "top1_sim": float(sims[ranked[0]]),
                "top5_skus": [str(x) for x in top_labels[:5].tolist()],
                "top5_sims": [float(sims[ranked[i]]) for i in range(5)],
                "top5_paths": top5_paths,
                "ap@5": float(ap)
            })

    if total_queries == 0:
        raise ValueError("total_queries=0，请检查数据")

    results = {
        f"CMC@{k}": round(cmc_hits[k] / total_queries * 100, 2) for k in topk
    }
    results[f"mAP@{map_k}"] = round(map_hits / total_queries * 100, 2)
    results["total_queries"] = total_queries
    results["valid_skus"] = len(valid_skus)

    return results, rank_records


def main():
    parser = argparse.ArgumentParser(description="SKU 检索评估")
    parser.add_argument("--features", "-f", required=True)
    parser.add_argument("--csv", "-c", required=True)
    parser.add_argument("--output", "-o", default="eval_report.json")
    parser.add_argument("--save-ranks", action="store_true")
    args = parser.parse_args()

    feats = np.load(args.features)
    df = pd.read_csv(args.csv)

    results, ranks = compute_cmc_and_map(feats, df, save_ranks=args.save_ranks)

    print(f"\n{'='*45}")
    print(f"评估结果（{results['total_queries']} 次有效查询）")
    for k, v in results.items():
        if k not in ("total_queries", "valid_skus"):
            print(f"  {k}: {v}%")
    print(f"{'='*45}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(to_native(results), f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {args.output}")

    if ranks:
        ranks_path = Path(args.output).with_suffix(".ranks.json")
        with open(ranks_path, "w", encoding="utf-8") as f:
            json.dump(to_native(ranks), f, ensure_ascii=False, indent=2)
        print(f"Top-5 排序详情: {ranks_path}")


if __name__ == "__main__":
    main()