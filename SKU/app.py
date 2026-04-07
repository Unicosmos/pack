"""
智能仓储识别系统 - Web演示界面
使用 Gradio 框架集成检测、特征提取、匹配与可视化功能

使用方法:
    python app.py
    
    # 指定端口
    python app.py --port 7860
    
    # 公开访问
    python app.py --share
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

import gradio as gr

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from ultralytics import YOLO


SKU_DB_DIR = "./sku_output"
YOLO_PATH = "./best.pt"
DEVICE = "cpu"

models_loaded = False
yolo_model = None
resnet_model = None
pca_model = None
sku_database = None
sku_features = None
sku_id_map = None
preprocess = None
feature_dim = 2048


def load_models():
    """加载所有模型和SKU库"""
    global yolo_model, resnet_model, pca_model, sku_database, sku_features, sku_id_map, preprocess, models_loaded, feature_dim
    
    if models_loaded:
        return
    
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    
    print("  [1/5] 加载 YOLO 模型...")
    yolo_model = YOLO(YOLO_PATH)
    
    print("  [2/5] 加载 ResNet50 模型...")
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_dim = resnet_model.fc.in_features
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model.to(device)
    resnet_model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    sku_db_path = Path(SKU_DB_DIR)
    
    print("  [3/5] 加载 SKU 数据库...")
    db_path = sku_db_path / "sku_database.json"
    if db_path.exists():
        with open(db_path, 'r', encoding='utf-8') as f:
            sku_database = json.load(f)
        print(f"        已加载 {len(sku_database.get('skus', []))} 个SKU")
    else:
        sku_database = {"skus": [], "metadata": {}}
        print("        警告: SKU数据库不存在")
    
    print("  [4/5] 加载 SKU 特征矩阵...")
    feat_path = sku_db_path / "sku_features.npy"
    if feat_path.exists():
        sku_features = np.load(feat_path)
        print(f"        特征矩阵形状: {sku_features.shape}")
    else:
        sku_features = np.array([])
        print("        警告: SKU特征矩阵不存在")
    
    print("  [5/5] 加载 PCA 模型...")
    pca_path = sku_db_path / "pca_model.joblib"
    if pca_path.exists():
        pca_model = joblib.load(pca_path)
        print(f"        PCA降维: {feature_dim} -> {pca_model.n_components_}")
    else:
        pca_model = None
        print("        警告: PCA模型不存在")
    
    if sku_database and "skus" in sku_database:
        sku_id_map = {i: sku["sku_id"] for i, sku in enumerate(sku_database["skus"])}
    else:
        sku_id_map = {}
    
    models_loaded = True
    print("=" * 60)
    print("模型加载完成!")
    print("=" * 60)


def detect_boxes(image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
    """YOLO检测箱体"""
    results = yolo_model.predict(
        source=image,
        conf=conf_threshold,
        device=DEVICE,
        verbose=False
    )
    
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        pred = results[0]
        for i in range(len(pred.boxes)):
            box = pred.boxes.xyxy[i].cpu().numpy()
            conf = float(pred.boxes.conf[i].cpu().numpy())
            cls_id = int(pred.boxes.cls[i].cpu().numpy())
            cls_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
            
            boxes.append({
                "bbox": list(map(int, box)),
                "confidence": conf,
                "class": cls_name
            })
    
    return boxes


def extract_features(image: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """提取箱体特征"""
    device = torch.device(DEVICE)
    
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image, mode='L').convert('RGB')
    elif image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')
    else:
        pil_image = Image.fromarray(image, mode='RGB')
    
    features = []
    
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_image.width, x2)
        y2 = min(pil_image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            features.append(np.zeros(feature_dim))
            continue
        
        cropped = pil_image.crop((x1, y1, x2, y2))
        
        input_tensor = preprocess(cropped).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = resnet_model(input_tensor).squeeze().cpu().numpy()
        
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    if len(features) > 0:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        features = features / norms
        
        if pca_model is not None:
            features = pca_model.transform(features)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            features = features / norms
    
    return features


def match_sku(features: np.ndarray, threshold: float = 0.85) -> List[Dict]:
    """匹配SKU"""
    if len(sku_features) == 0:
        return [{"sku_id": "Unknown", "similarity": 0.0, "status": "no_sku"} for _ in range(len(features))]
    
    similarities = np.dot(features, sku_features.T)
    
    results = []
    for i in range(len(features)):
        sim_scores = similarities[i]
        max_idx = np.argmax(sim_scores)
        max_sim = float(sim_scores[max_idx])
        
        if max_sim >= threshold:
            sku_id = sku_id_map.get(max_idx, "Unknown")
            results.append({
                "sku_id": sku_id,
                "similarity": max_sim,
                "status": "matched"
            })
        else:
            results.append({
                "sku_id": "Unknown",
                "similarity": max_sim,
                "status": "unmatched"
            })
    
    return results


def draw_results(
    image: np.ndarray,
    boxes: List[Dict],
    match_results: List[Dict]
) -> np.ndarray:
    """绘制检测结果"""
    result_img = image.copy()
    
    matched_count = sum(1 for r in match_results if r["status"] == "matched")
    total_count = len(boxes)
    
    cv2.rectangle(result_img, (5, 5), (350, 35), (0, 0, 0), -1)
    cv2.putText(
        result_img,
        f"Total: {total_count}  Matched: {matched_count}  Unknown: {total_count - matched_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    for box, result in zip(boxes, match_results):
        x1, y1, x2, y2 = box["bbox"]
        
        if result["status"] == "matched":
            color = (0, 255, 0)
            label = f"{result['sku_id']} ({result['similarity']:.2f})"
        else:
            color = (0, 0, 255)
            label = f"Unknown ({result['similarity']:.2f})"
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(result_img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(
            result_img,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    return result_img


def visualize_features(
    features: np.ndarray,
    match_results: List[Dict]
) -> plt.Figure:
    """特征空间可视化"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(sku_features) > 0:
        n_bg = min(500, len(sku_features))
        if len(sku_features) > n_bg:
            indices = np.random.choice(len(sku_features), n_bg, replace=False)
            bg_features = sku_features[indices]
        else:
            bg_features = sku_features
        
        if len(bg_features) > 1:
            perplexity = min(30, max(5, len(bg_features) // 4))
            
            if len(features) > 0:
                all_features = np.vstack([bg_features, features])
            else:
                all_features = bg_features
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
            all_2d = tsne.fit_transform(all_features)
            
            bg_2d = all_2d[:len(bg_features)]
            new_2d = all_2d[len(bg_features):]
            
            ax.scatter(
                bg_2d[:, 0], bg_2d[:, 1],
                c='lightgray', s=20, alpha=0.5, label='SKU Library'
            )
            
            if len(new_2d) > 0:
                colors = ['green' if r["status"] == "matched" else 'red' for r in match_results]
                ax.scatter(
                    new_2d[:, 0], new_2d[:, 1],
                    c=colors, s=100, alpha=0.8, edgecolors='black', linewidths=1.5,
                    label='Detected Boxes'
                )
                
                for i, (x, y) in enumerate(new_2d):
                    ax.annotate(f"#{i+1}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    else:
        ax.text(0.5, 0.5, "SKU库为空，无法生成特征可视化",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    ax.set_title('Feature Space Visualization (t-SNE)', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def parse_annotations(annotation_text: str, image_width: int = 0, image_height: int = 0) -> List[Dict]:
    """
    解析标注信息
    
    支持格式:
    1. JSON数组: [{"bbox": [x1,y1,x2,y2], "label": "box1"}, ...]
    2. 分类ID+坐标格式: class_id x1 y1 x2 y2 (每行一个框)
    3. 分类ID+坐标格式(逗号分隔): class_id,x1,y1,x2,y2
    4. 简单格式: x1,y1,x2,y2 (每行一个框)
    5. COCO格式: class_id center_x center_y width height (归一化坐标)
    
    Args:
        annotation_text: 标注文本
        image_width: 图像宽度（用于 COCO 格式转换）
        image_height: 图像高度（用于 COCO 格式转换）
        
    Returns:
        boxes: 标注框列表
    """
    boxes = []
    
    annotation_text = annotation_text.strip()
    
    if not annotation_text:
        return boxes
    
    try:
        data = json.loads(annotation_text)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    if "bbox" in item:
                        bbox = item["bbox"]
                        if len(bbox) == 4:
                            boxes.append({
                                "bbox": list(map(int, bbox)),
                                "label": item.get("label", f"box_{i+1}")
                            })
                    elif all(k in item for k in ["x1", "y1", "x2", "y2"]):
                        boxes.append({
                            "bbox": [int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])],
                            "label": item.get("label", f"box_{i+1}")
                        })
        return boxes
    
    except json.JSONDecodeError:
        pass
    
    for line in annotation_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 尝试按空格分割（分类ID+坐标格式或 COCO 格式）
        parts = line.split()
        if len(parts) >= 5:
            try:
                class_id = parts[0]
                coords = list(map(float, parts[1:5]))
                
                # 检查是否为 COCO 格式（值在 0-1 之间）
                if all(0 <= x <= 1 for x in coords) and image_width > 0 and image_height > 0:
                    # COCO 格式：class_id center_x center_y width height
                    center_x, center_y, width, height = coords
                    x1 = int((center_x - width/2) * image_width)
                    y1 = int((center_y - height/2) * image_height)
                    x2 = int((center_x + width/2) * image_width)
                    y2 = int((center_y + height/2) * image_height)
                    bbox = [x1, y1, x2, y2]
                else:
                    # 普通格式：class_id x1 y1 x2 y2
                    bbox = list(map(int, coords))
                
                boxes.append({
                    "bbox": bbox,
                    "label": f"class_{class_id}"
                })
                continue
            except ValueError:
                pass
        
        # 尝试按逗号分割（分类ID+坐标格式或简单格式）
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                if len(parts) >= 5:
                    # 分类ID+坐标格式
                    class_id = parts[0]
                    coords = list(map(float, parts[1:5]))
                    
                    # 检查是否为 COCO 格式（值在 0-1 之间）
                    if all(0 <= x <= 1 for x in coords) and image_width > 0 and image_height > 0:
                        # COCO 格式：class_id,center_x,center_y,width,height
                        center_x, center_y, width, height = coords
                        x1 = int((center_x - width/2) * image_width)
                        y1 = int((center_y - height/2) * image_height)
                        x2 = int((center_x + width/2) * image_width)
                        y2 = int((center_y + height/2) * image_height)
                        bbox = [x1, y1, x2, y2]
                    else:
                        # 普通格式：class_id,x1,y1,x2,y2
                        bbox = list(map(int, coords))
                    
                    boxes.append({
                        "bbox": bbox,
                        "label": f"class_{class_id}"
                    })
                else:
                    # 简单格式
                    coords = list(map(int, parts[:4]))
                    boxes.append({
                        "bbox": coords,
                        "label": f"box_{len(boxes)+1}"
                    })
            except ValueError:
                continue
    
    return boxes


def draw_annotation_preview(
    image: np.ndarray,
    boxes: List[Dict]
) -> np.ndarray:
    """
    绘制标注预览图
    
    Args:
        image: 原始图片
        boxes: 标注框列表
        
    Returns:
        preview_img: 带标注的预览图
    """
    preview_img = image.copy()
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        color = colors[i % len(colors)]
        
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 3)
        
        label = box.get("label", f"#{i+1}")
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(preview_img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(
            preview_img,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    return preview_img


def add_to_sku_database(
    image: np.ndarray,
    boxes: List[Dict],
    sku_id: str,
    sku_name: str = ""
) -> Tuple[bool, str]:
    """
    将标注的箱体添加到SKU库
    
    Args:
        image: 原始图片
        boxes: 标注框列表
        sku_id: SKU ID
        sku_name: SKU 名称
        
    Returns:
        success: 是否成功
        message: 结果消息
    """
    global sku_database, sku_features, sku_id_map
    
    if not sku_id or not sku_id.strip():
        return False, "SKU ID 不能为空"
    
    sku_id = sku_id.strip()
    
    if len(boxes) == 0:
        return False, "没有有效的标注框"
    
    load_models()
    
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image, mode='L').convert('RGB')
    elif image.shape[2] == 4:
        pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')
    else:
        pil_image = Image.fromarray(image, mode='RGB')
    
    new_features = []
    new_boxes_info = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_image.width, x2)
        y2 = min(pil_image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        cropped = pil_image.crop((x1, y1, x2, y2))
        
        device = torch.device(DEVICE)
        input_tensor = preprocess(cropped).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = resnet_model(input_tensor).squeeze().cpu().numpy()
        
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        
        if pca_model is not None:
            feat = pca_model.transform(feat.reshape(1, -1)).squeeze()
            feat = feat / (np.linalg.norm(feat) + 1e-8)
        
        new_features.append(feat)
        new_boxes_info.append({
            "bbox": box["bbox"],
            "label": box.get("label", f"box_{i+1}"),
            "width": x2 - x1,
            "height": y2 - y1
        })
    
    if len(new_features) == 0:
        return False, "没有有效的特征被提取"
    
    new_features = np.array(new_features, dtype=np.float32)
    
    if sku_features is None or len(sku_features) == 0:
        sku_features = new_features
    else:
        sku_features = np.vstack([sku_features, new_features])
    
    if sku_database is None:
        sku_database = {"skus": [], "metadata": {}}
    
    sku_entry = {
        "sku_id": sku_id,
        "sku_name": sku_name if sku_name else sku_id,
        "box_count": len(new_boxes_info),
        "boxes": new_boxes_info,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    existing_idx = None
    for i, sku in enumerate(sku_database.get("skus", [])):
        if sku.get("sku_id") == sku_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        old_count = sku_database["skus"][existing_idx].get("box_count", 0)
        sku_entry["box_count"] = old_count + len(new_boxes_info)
        sku_entry["boxes"] = sku_database["skus"][existing_idx].get("boxes", []) + new_boxes_info
        sku_entry["created_at"] = sku_database["skus"][existing_idx].get("created_at", sku_entry["created_at"])
        sku_database["skus"][existing_idx] = sku_entry
        message = f"已更新 SKU '{sku_id}'，新增 {len(new_boxes_info)} 个箱体，总计 {sku_entry['box_count']} 个"
    else:
        sku_database["skus"].append(sku_entry)
        message = f"已添加新 SKU '{sku_id}'，包含 {len(new_boxes_info)} 个箱体"
    
    sku_database["metadata"] = {
        "total_skus": len(sku_database["skus"]),
        "total_boxes": int(sku_features.shape[0]),
        "feature_dim": sku_features.shape[1],
        "last_updated": datetime.now().isoformat()
    }
    
    sku_id_map = {i: sku["sku_id"] for i, sku in enumerate(sku_database["skus"])}
    
    sku_db_path = Path(SKU_DB_DIR)
    sku_db_path.mkdir(parents=True, exist_ok=True)
    
    with open(sku_db_path / "sku_database.json", 'w', encoding='utf-8') as f:
        json.dump(sku_database, f, indent=2, ensure_ascii=False)
    
    np.save(sku_db_path / "sku_features.npy", sku_features)
    
    return True, message


def get_sku_database_info() -> str:
    """
    获取SKU库信息
    
    Returns:
        info: SKU库信息文本
    """
    if sku_database is None or len(sku_database.get("skus", [])) == 0:
        return "📦 SKU库为空"
    
    info_lines = [
        f"📊 **SKU库统计**",
        f"- 总SKU数: {len(sku_database['skus'])}",
        f"- 总箱体数: {sku_database['metadata'].get('total_boxes', 0)}",
        f"- 特征维度: {sku_database['metadata'].get('feature_dim', 'N/A')}",
        f"- 最后更新: {sku_database['metadata'].get('last_updated', 'N/A')}",
        "",
        "📋 **SKU列表:**"
    ]
    
    for sku in sku_database["skus"]:
        info_lines.append(
            f"  - **{sku['sku_id']}**: {sku.get('sku_name', sku['sku_id'])} "
            f"({sku.get('box_count', 0)} 个箱体)"
        )
    
    return "\n".join(info_lines)


def process_and_visualize(
    input_image: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, plt.Figure, str]:
    """
    核心处理函数
    
    Args:
        input_image: 输入图片 (numpy array)
        threshold: 匹配阈值
        
    Returns:
        result_image: 结果图片
        feature_fig: 特征可视化图
        json_result: JSON格式结果
    """
    if input_image is None:
        empty_fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "请上传图片", ha='center', va='center', fontsize=14)
        return None, empty_fig, "{}"
    
    if len(sku_features) == 0:
        result_img = input_image.copy()
        cv2.putText(result_img, "SKU库为空，请先运行聚类模块", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        empty_fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "SKU库为空", ha='center', va='center', fontsize=14)
        return result_img, empty_fig, '{"error": "SKU库为空"}'
    
    boxes = detect_boxes(input_image, conf_threshold=0.5)
    
    if len(boxes) == 0:
        result_img = input_image.copy()
        cv2.putText(result_img, "未检测到箱体", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        empty_fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "未检测到箱体", ha='center', va='center', fontsize=14)
        return result_img, empty_fig, '{"total_boxes": 0}'
    
    features = extract_features(input_image, boxes)
    
    match_results = match_sku(features, threshold)
    
    result_img = draw_results(input_image, boxes, match_results)
    
    feature_fig = visualize_features(features, match_results)
    
    matched_count = sum(1 for r in match_results if r["status"] == "matched")
    
    result = {
        "total_boxes": len(boxes),
        "matched_boxes": matched_count,
        "unmatched_boxes": len(boxes) - matched_count,
        "threshold": threshold,
        "details": [
            {
                "box_id": i + 1,
                "bbox": box["bbox"],
                "sku_id": result["sku_id"],
                "similarity": round(result["similarity"], 4),
                "status": result["status"]
            }
            for i, (box, result) in enumerate(zip(boxes, match_results))
        ]
    }
    
    json_result = json.dumps(result, indent=2, ensure_ascii=False)
    
    return result_img, feature_fig, json_result


def build_interface():
    """构建Gradio界面"""
    
    with gr.Blocks() as demo:
        demo.title = "仓储物品智能识别系统"
        
        gr.Markdown(
            """
            # 📦 仓储物品智能识别系统
            **基于 YOLOv8 + ResNet50 的地堆箱体检测与 SKU 匹配**
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("🔍 箱体检测"):
                gr.Markdown("上传图片后，系统将自动检测箱体并匹配SKU库中的商品。")
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 📤 输入")
                        
                        input_image = gr.Image(
                            label="上传图片",
                            type="numpy",
                            height=400
                        )
                        
                        threshold = gr.Slider(
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                            label="匹配阈值",
                            info="相似度超过阈值才会匹配SKU"
                        )
                        
                        with gr.Row():
                            detect_btn = gr.Button(
                                "🔍 开始检测",
                                variant="primary",
                                size="lg"
                            )
                            clear_btn = gr.Button(
                                "🗑️ 清除",
                                variant="secondary",
                                size="lg"
                            )
                        
                        gr.Markdown(
                            """
                            ---
                            ### 📊 SKU库状态
                            """
                        )
                        
                        if sku_database and "metadata" in sku_database:
                            sku_info = f"""
                            - SKU数量: **{sku_database['metadata'].get('total_skus', 0)}**
                            - 总箱体数: **{sku_database['metadata'].get('total_boxes', 0)}**
                            - 特征维度: **{sku_database['metadata'].get('feature_dim', 'N/A')}**
                            """
                        else:
                            sku_info = "⚠️ SKU库未加载"
                        
                        gr.Markdown(sku_info)
                    
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("### 📥 输出")
                        
                        with gr.Tabs():
                            with gr.TabItem("🔍 检测结果"):
                                result_image = gr.Image(
                                    label="检测结果",
                                    type="numpy",
                                    height=500,
                                    elem_classes=["output-image"]
                                )
                            
                            with gr.TabItem("📊 特征可视化"):
                                feature_plot = gr.Plot(
                                    label="特征空间可视化"
                                )
                            
                            with gr.TabItem("📋 详细数据"):
                                json_output = gr.Code(
                                    label="JSON结果",
                                    language="json",
                                    lines=20
                                )
                
                detect_btn.click(
                    fn=process_and_visualize,
                    inputs=[input_image, threshold],
                    outputs=[result_image, feature_plot, json_output]
                )
                
                clear_btn.click(
                    fn=lambda: (None, None, None, "{}"),
                    inputs=[],
                    outputs=[input_image, result_image, feature_plot, json_output]
                )
            
            with gr.TabItem("📦 SKU库管理"):
                gr.Markdown("手动更新SKU库，导入图片和标注，提取特征并保存到数据库。")
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### 📤 输入")
                        
                        sku_input_image = gr.Image(
                            label="上传图片",
                            type="numpy",
                            height=400
                        )
                        
                        sku_id_input = gr.Textbox(
                            label="SKU ID",
                            placeholder="例如: SKU_001",
                            info="必填，唯一标识符"
                        )
                        
                        sku_name_input = gr.Textbox(
                            label="SKU 名称",
                            placeholder="例如: 可口可乐330ml",
                            info="可选，商品名称"
                        )
                        
                        gr.Markdown("**标注格式说明:**")
                        gr.Markdown(
                            """
                            - **COCO格式** (推荐): `class_id center_x center_y width height` (归一化坐标，值在0-1之间)
                            - **分类ID+坐标格式**: `class_id x1 y1 x2 y2` (每行一个框，像素坐标)
                            - **分类ID+坐标格式** (逗号分隔): `class_id,x1,y1,x2,y2` (像素坐标)
                            - **简单格式**: `x1,y1,x2,y2` (每行一个框，像素坐标)
                            - **JSON数组**: `[{\"bbox\": [x1,y1,x2,y2], \"label\": \"box1\"}, ...]` (像素坐标)
                            """
                        )
                        
                        annotation_input = gr.Code(
                            label="标注信息",
                            language="json",
                            lines=10
                        )
                        
                        with gr.Row():
                            preview_btn = gr.Button(
                                "👁️ 预览标注",
                                variant="secondary",
                                size="lg"
                            )
                            add_sku_btn = gr.Button(
                                "✅ 添加到SKU库",
                                variant="primary",
                                size="lg"
                            )
                        
                        clear_sku_btn = gr.Button(
                            "🗑️ 清除",
                            variant="secondary"
                        )
                    
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### 📥 输出")
                        
                        annotation_preview = gr.Image(
                            label="标注预览",
                            type="numpy",
                            height=400
                        )
                        
                        sku_status = gr.Textbox(
                            label="操作状态",
                            lines=3,
                            interactive=False
                        )
                        
                        gr.Markdown("### 📊 SKU库信息")
                        sku_db_info = gr.Markdown(
                            value=get_sku_database_info()
                        )
                        
                        refresh_btn = gr.Button(
                            "🔄 刷新SKU库信息",
                            variant="secondary"
                        )
                
                def preview_annotation(image, annotation_text):
                    if image is None:
                        return None, "请先上传图片"
                    
                    # 解析标注，传入图像宽高以支持 COCO 格式
                    boxes = parse_annotations(annotation_text, image.shape[1], image.shape[0])
                    
                    if len(boxes) == 0:
                        return image.copy(), "未检测到有效标注"
                    
                    preview_img = draw_annotation_preview(image, boxes)
                    return preview_img, f"已解析 {len(boxes)} 个标注框"
                
                def add_sku_handler(image, annotation_text, sku_id, sku_name):
                    if image is None:
                        return None, "❌ 请先上传图片", get_sku_database_info()
                    
                    # 解析标注，传入图像宽高以支持 COCO 格式
                    boxes = parse_annotations(annotation_text, image.shape[1], image.shape[0])
                    
                    if len(boxes) == 0:
                        return None, "❌ 未检测到有效标注", get_sku_database_info()
                    
                    success, message = add_to_sku_database(image, boxes, sku_id, sku_name)
                    
                    preview_img = draw_annotation_preview(image, boxes) if success else None
                    
                    status_icon = "✅" if success else "❌"
                    
                    return preview_img, f"{status_icon} {message}", get_sku_database_info()
                
                preview_btn.click(
                    fn=preview_annotation,
                    inputs=[sku_input_image, annotation_input],
                    outputs=[annotation_preview, sku_status]
                )
                
                add_sku_btn.click(
                    fn=add_sku_handler,
                    inputs=[sku_input_image, annotation_input, sku_id_input, sku_name_input],
                    outputs=[annotation_preview, sku_status, sku_db_info]
                )
                
                clear_sku_btn.click(
                    fn=lambda: (None, "", "", "", None, "", get_sku_database_info()),
                    inputs=[],
                    outputs=[sku_input_image, sku_id_input, sku_name_input, annotation_input, 
                            annotation_preview, sku_status, sku_db_info]
                )
                
                refresh_btn.click(
                    fn=get_sku_database_info,
                    inputs=[],
                    outputs=[sku_db_info]
                )
        
        gr.Markdown(
            """
            ---
            ### 📝 使用说明
            
            **🔍 箱体检测页面:**
            1. 上传图片: 支持拖拽或点击上传
            2. 调整阈值: 阈值越高，匹配越严格
            3. 开始检测: 点击按钮开始处理
            4. 查看结果: 绿色框=已匹配, 红色框=未知
            
            **📦 SKU库管理页面:**
            1. 上传图片: 包含需要添加的箱体
            2. 输入SKU ID: 唯一标识符（必填）
            3. 输入SKU名称: 商品名称（可选）
            4. 输入标注: JSON格式或简单格式
            5. 预览标注: 查看标注是否正确
            6. 添加到SKU库: 提取特征并保存
            """
        )
    
    return demo


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='智能仓储识别系统 Web界面')
    parser.add_argument('--port', type=int, default=7860, help='端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    parser.add_argument('--share', action='store_true', help='创建公开链接')
    parser.add_argument('--sku-db', type=str, default='./sku_output', help='SKU库目录')
    parser.add_argument('--yolo', type=str, default='./best.pt', help='YOLO模型路径')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    return parser.parse_args()


def main():
    """主函数"""
    global SKU_DB_DIR, YOLO_PATH, DEVICE
    
    args = parse_arguments()
    
    SKU_DB_DIR = args.sku_db
    YOLO_PATH = args.yolo
    DEVICE = args.device
    
    load_models()
    
    demo = build_interface()
    
    print(f"\n启动 Web 服务...")
    print(f"地址: http://{args.host}:{args.port}")
    if args.share:
        print("公开链接已启用")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == '__main__':
    main()
