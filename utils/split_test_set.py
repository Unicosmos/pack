"""
从训练集划分测试集的脚本
保持验证集不变，从训练集划分指定比例作为测试集
"""

import os
import random
import shutil
import argparse
"""
# 划分测试集
python utils/split_test_set.py --ratio 0.05 --seed 0

# 还原测试集
python utils/split_test_set.py --restore"""

def split_test_set(
    dataset_path: str,
    test_ratio: float = 0.05,
    seed: int = 0,
    copy_mode: bool = False
):
    """
    从训练集划分测试集
    
    Args:
        dataset_path: YOLO数据集路径
        test_ratio: 测试集比例
        seed: 随机种子
        copy_mode: True为复制，False为移动
    """
    random.seed(seed)
    
    train_img_dir = os.path.join(dataset_path, "images/train")
    train_label_dir = os.path.join(dataset_path, "labels/train")
    test_img_dir = os.path.join(dataset_path, "images/test")
    test_label_dir = os.path.join(dataset_path, "labels/test")
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"原训练集图片数: {len(train_images)}")
    
    test_count = int(len(train_images) * test_ratio)
    print(f"划分测试集数量: {test_count} ({test_ratio*100:.0f}%)")
    
    test_images = random.sample(train_images, test_count)
    
    operation = shutil.copy2 if copy_mode else shutil.move
    op_name = "复制" if copy_mode else "移动"
    
    for img_name in test_images:
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(test_img_dir, img_name)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(train_label_dir, label_name)
        dst_label = os.path.join(test_label_dir, label_name)
        
        operation(src_img, dst_img)
        if os.path.exists(src_label):
            operation(src_label, dst_label)
    
    print(f"\n{op_name}完成!")
    print(f"训练集图片数: {len(os.listdir(train_img_dir))}")
    print(f"验证集图片数: {len(os.listdir(os.path.join(dataset_path, 'images/val')))}")
    print(f"测试集图片数: {len(os.listdir(test_img_dir))}")


def restore_test_set(dataset_path: str):
    """将测试集还原回训练集"""
    train_img_dir = os.path.join(dataset_path, "images/train")
    train_label_dir = os.path.join(dataset_path, "labels/train")
    test_img_dir = os.path.join(dataset_path, "images/test")
    test_label_dir = os.path.join(dataset_path, "labels/test")
    
    if not os.path.exists(test_img_dir):
        print("测试集目录不存在，无需还原")
        return
    
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"测试集图片数: {len(test_images)}")
    
    for img_name in test_images:
        src_img = os.path.join(test_img_dir, img_name)
        dst_img = os.path.join(train_img_dir, img_name)
        shutil.move(src_img, dst_img)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(test_label_dir, label_name)
        dst_label = os.path.join(train_label_dir, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
    
    print(f"\n已还原!")
    print(f"训练集图片数: {len(os.listdir(train_img_dir))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从训练集划分测试集")
    parser.add_argument("--dataset", type=str, 
                        default="/root/source/data2/hyg/projects/pack/coco_style_oneclass/yolo_dataset_seg",
                        help="YOLO数据集路径")
    parser.add_argument("--ratio", type=float, default=0.05, help="测试集比例")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--copy", action="store_true", help="复制模式（默认移动）")
    parser.add_argument("--restore", action="store_true", help="还原测试集到训练集")
    
    args = parser.parse_args()
    
    if args.restore:
        restore_test_set(args.dataset)
    else:
        split_test_set(args.dataset, args.ratio, args.seed, args.copy)
