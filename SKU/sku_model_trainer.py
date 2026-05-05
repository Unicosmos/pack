from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import torch
import random
import numpy as np
import argparse
from pathlib import Path

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import HardTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler

def main():
    parser = argparse.ArgumentParser(description='SKU模型训练')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4, help='每类的样本数')
    parser.add_argument('--n_labels', type=int, default=4, help='每批次包含的类别数')
    parser.add_argument('--data_dir', type=str, default="../sku_library", help='数据集目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    # 加载并准备数据
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")

    # OML 需要的列名
    df_train = df_train.rename(columns={"image_name": "path"})
    df_val = df_val.rename(columns={"image_name": "path"})

    # 构建图片路径，图片在 sku_library/images/{sku_id} 里
    images_dir = Path("../sku_library/images")
    df_train["path"] = df_train.apply(lambda x: str(images_dir / f"{int(x['sku_id']):06d}" / x["path"]), axis=1)
    df_val["path"] = df_val.apply(lambda x: str(images_dir / f"{int(x['sku_id']):06d}" / x["path"]), axis=1)
    
    # 把原来的label列转换成唯一整数
    all_labels = pd.concat([df_train["label"], df_val["label"]]).unique()
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    
    df_train["label"] = df_train["label"].map(label_to_id)
    df_val["label"] = df_val["label"].map(label_to_id)
    
    # 给验证集加上查询/图库列，简化处理
    df_val["is_query"] = True
    df_val["is_gallery"] = True

    train = d.ImageLabeledDataset(df_train, transform=transform)
    val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(val)}")
    print(f"训练配置: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, n_labels={args.n_labels}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = TripletLossWithMiner(0.1, HardTripletsMiner(), need_logs=True)
    sampler = BalanceSampler(train.get_labels(), n_labels=args.n_labels, n_instances=args.batch_size)

    print("=" * 50)
    print(f"开始训练，共 {args.epochs} 轮")
    print("=" * 50)

    # 创建保存目录
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss = 0
        batch_count = 0

        # 训练
        model.train()
        for batch in DataLoader(train, batch_sampler=sampler):
            embeddings = model(batch["input_tensors"])
            loss = criterion(embeddings, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            batch_count += 1
            print(f"  Batch loss: {loss.item():.4f} | Avg loss: {epoch_loss/batch_count:.4f}")

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} 完成，平均Loss: {avg_loss:.4f}")
        
        # 每轮验证
        model.eval()
        embeddings = inference(model, val, batch_size=4, num_workers=0)
        rr = RetrievalResults.from_embeddings(embeddings, val, n_items=5)
        rr = AdaptiveThresholding(n_std=2).process(rr)
        metrics = calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,))
        print(f"\nEpoch {epoch+1} 验证指标:")
        print(metrics)

    print("\n" + "=" * 50)
    print("训练完成，保存模型...")
    print("=" * 50)

    save_path = save_dir / "sku_trained_vits16_dino.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    main()
