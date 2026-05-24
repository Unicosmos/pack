#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import time, copy, random          # ← 新增 random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from oml.datasets.base import DatasetWithLabels
except ImportError:
    from oml.datasets import DatasetWithLabels

try:
    from oml.models import ViTExtractor
except ImportError:
    from oml.models.vit import ViTExtractor

try:
    from oml.losses.triplet import TripletLossWithMiner
except ImportError:
    from oml.losses import TripletLossWithMiner

try:
    from oml.miners import AllTripletsMiner
except ImportError:
    try:
        from oml.miners.inbatch_all_triplets import AllTripletsMiner
    except ImportError:
        from oml.miners.inbatch import AllTripletsMiner

try:
    from oml.samplers.balance import BalanceSampler
except ImportError:
    from oml.samplers import BalanceSampler


class L2NormalizedViT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def extract_features(model, df, data_dir, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    paths = df["path"].apply(lambda p: str(data_dir / p)).tolist()
    features = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            images = []
            for p in batch_paths:
                from PIL import Image
                img = Image.open(p).convert("RGB")
                images.append(transform(img))
            
            batch = torch.stack(images).to(device)
            emb = model(batch)
            features.append(emb.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def compute_ap_at_k(retrieved_labels, query_label, k):
    precisions = []
    num_hits = 0
    for i, label in enumerate(retrieved_labels[:k], 1):
        if label == query_label:
            num_hits += 1
            precisions.append(num_hits / i)
    
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def compute_metrics(features, df, topk=(1, 5), map_k=5):
    n = len(df)
    assert features.shape[0] == n

    base_mask = (df['is_base'] == 1) | (df['is_base'] == '1') | (df['is_base'] == 1.0)
    base_positions = np.where(base_mask)[0]
    if len(base_positions) == 0:
        base_mask = df['is_base'].astype(str) == '1'
        base_positions = np.where(base_mask)[0]

    base_skus = df.iloc[base_positions]['sku_id'].values
    from collections import Counter
    sku_base_counts = Counter(base_skus)
    valid_skus = {sku for sku, cnt in sku_base_counts.items() if cnt >= 2}

    if not valid_skus:
        return None

    all_sims = features @ features.T
    parents = df['parent'].fillna("").astype(str).values
    sku_ids = df['sku_id'].values

    cmc_hits = {k: 0 for k in topk}
    map_sum = 0
    total_queries = 0

    for q_pos in base_positions:
        q_sku = sku_ids[q_pos]
        if q_sku not in valid_skus:
            continue

        q_filename = Path(df.iloc[q_pos]['path']).name

        mask = np.ones(n, dtype=bool)
        mask[q_pos] = False
        mask &= (parents != q_filename)

        sims = all_sims[q_pos, mask]
        gallery_labels = sku_ids[mask]
        ranked = np.argsort(-sims)
        top_labels = gallery_labels[ranked]

        correct_positions = np.where(top_labels == q_sku)[0]
        if len(correct_positions) == 0:
            continue

        first_hit = int(correct_positions[0]) + 1

        for k in topk:
            if first_hit <= k:
                cmc_hits[k] += 1

        ap = compute_ap_at_k(top_labels, q_sku, map_k)
        map_sum += ap
        total_queries += 1

    if total_queries == 0:
        return None

    results = {
        f"CMC@{k}": round(cmc_hits[k] / total_queries * 100, 2) for k in topk
    }
    results[f"mAP@{map_k}"] = round(map_sum / total_queries * 100, 2)
    results["total_queries"] = total_queries
    results["valid_skus"] = len(valid_skus)
    return results


def main():
    parser = argparse.ArgumentParser(description="SKU 度量学习微调")
    parser.add_argument("--input", "-i", required=True, help="sku_library 目录路径")
    parser.add_argument("--train-csv", default="train.csv", help="训练集 CSV")
    parser.add_argument("--test-csv", default="test.csv", help="测试集 CSV（用于验证）")
    parser.add_argument("--output", "-o", default="vits16_dino_finetuned.pth", help="输出权重路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="每类样本数")
    parser.add_argument("--n-labels", "-l", type=int, default=4, help="每批次类别数")
    parser.add_argument("--margin", "-m", type=float, default=0.1, help="三元组 margin")
    parser.add_argument("--device", default="cpu", help="cuda / cpu / auto")
    parser.add_argument("--unfreeze-last", type=int, default=2, help="解冻最后 N 层")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心")
    parser.add_argument("--metric", default="mAP", choices=["CMC", "mAP"], help="早停指标")
    parser.add_argument("--seed", type=int, default=0, help="全局随机种子")   # ← 新增参数
    args = parser.parse_args()

    # ========== 全局随机种子固定（保证可复现）==========
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定: {seed}")
    # ================================================

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    print(f"设备: {device}")

    data_dir = Path(args.input)

    train_csv = data_dir / args.train_csv
    if not train_csv.exists():
        raise FileNotFoundError(f"找不到训练集: {train_csv}")
    df_train = pd.read_csv(train_csv)
    df_train["path"] = df_train["path"].apply(lambda p: str(data_dir / p))
    print(f"训练集: {len(df_train)} 张图, {df_train['sku_id'].nunique()} 个 SKU")

    test_csv = data_dir / args.test_csv
    if not test_csv.exists():
        raise FileNotFoundError(f"找不到测试集: {test_csv}")
    df_test = pd.read_csv(test_csv)
    df_test["path"] = df_test["path"].apply(lambda p: str(data_dir / p))
    print(f"测试集: {len(df_test)} 张图, {df_test['sku_id'].nunique()} 个 SKU")

    # ========== 在线增强 ==========
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.ColorJitter(0.05, 0.05)], p=0.3),      # ← 减弱
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.3))], p=0.2),  # ← 减弱
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # ===========================================

    dataset = DatasetWithLabels(df_train, transform=train_transform)

    sampler = BalanceSampler(
        labels=dataset.get_labels(),
        n_labels=args.n_labels,
        n_instances=args.batch_size
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    print("加载预训练模型: vits16_dino")
    base_model = ViTExtractor.from_pretrained("vits16_dino").to(device)
    model = L2NormalizedViT(base_model)

    total_blocks = len(base_model.model.blocks)
    freeze_until = total_blocks - args.unfreeze_last
    for i, block in enumerate(base_model.model.blocks):
        for param in block.parameters():
            param.requires_grad = (i >= freeze_until)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"总参数: {trainable + frozen:,}")
    print(f"可训练: {trainable:,} (最后 {args.unfreeze_last} 层)")
    print(f"冻结:   {frozen:,}")

    criterion = TripletLossWithMiner(
        margin=args.margin,
        miner=AllTripletsMiner()
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    best_metric = 0.0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    best_results = None

    print(f"\n开始训练: {args.epochs} epochs (早停耐心={args.patience}, 指标={args.metric})")
    print("=" * 50)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in loader:
            images = batch["input_tensors"].to(device)
            labels = batch["labels"].long().to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        print(f"\nEpoch [{epoch+1}/{args.epochs}] 训练 Loss: {avg_loss:.4f}")

        print("  测试集验证中...")
        test_feats = extract_features(model, df_test, data_dir, device)
        test_results = compute_metrics(test_feats, df_test, topk=(1, 5), map_k=5)
        
        if test_results is None:
            print("  [WARN] 测试集无有效查询")
            continue

        print(f"  测试集 CMC@1: {test_results['CMC@1']:.2f}% | CMC@5: {test_results['CMC@5']:.2f}% | mAP@5: {test_results['mAP@5']:.2f}%")

        current_metric = test_results['mAP@5'] if args.metric == "mAP" else test_results['CMC@1']
        
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            # ========== 深拷贝到 CPU，避免被后续训练污染 ==========
            best_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
            # ===================================================
            best_epoch = epoch + 1
            best_results = test_results.copy()
            print(f"  → 🎉 新的最佳 {args.metric}: {best_metric:.2f}% (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"  → {args.metric} 未提升 ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n{'='*50}")
            print(f"早停触发！连续 {args.patience} 轮测试集未提升")
            print(f"最佳结果: Epoch {best_epoch}")
            for k, v in best_results.items():
                if k not in ("total_queries", "valid_skus"):
                    print(f"  {k}: {v:.2f}%")
            print(f"{'='*50}")
            break

    # ========== 直接保存最佳权重，不再 load 回去 ==========
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_state is not None:
        torch.save(best_state, output_path)
        print(f"\n已保存最佳权重 (Epoch {best_epoch}) → {output_path}")
    else:
        torch.save(base_model.state_dict(), output_path)
        print(f"\n已保存最终权重 → {output_path}")
    # ===================================================

    # ========== 加载最佳权重回内存，确保最终评估一致 ==========
    if best_state is not None:
        base_model.load_state_dict(best_state)
        print(f"已加载最佳权重 (Epoch {best_epoch}) 回内存")
    # =========================================================
    
    # 最终测试集评估（用当前内存里的模型，如果是早停就是最佳模型）
    print(f"\n{'='*50}")
    print("最终测试集评估:")
    final_feats = extract_features(model, df_test, data_dir, device)
    final_results = compute_metrics(final_feats, df_test, topk=(1, 5), map_k=5)
    if final_results:
        for k, v in final_results.items():
            if k not in ("total_queries", "valid_skus"):
                print(f"  {k}: {v:.2f}%")
    print(f"{'='*50}")

    # ========== 训练时间 ==========
    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed:.1f}s)")
    # =============================


if __name__ == "__main__":
    main()