# YOLO训练复现规划

## 概述

本规划基于《YOLO恢复\_复现指南.md》，分阶段完成YOLO纸箱分割项目的完整复现。

***

## 阶段一：环境准备

### 1.1 检查当前环境状态

- [ ] 检查Python版本（建议3.8+）
- [ ] 检查CUDA版本（建议11.8）
- [ ] 检查虚拟环境是否存在

### 1.2 安装依赖

- [ ] 安装核心依赖：`pip install ultralytics scipy opencv-python tqdm pyyaml python-dotenv`
- [ ] 安装MLflow：`pip install mlflow`
- [ ] 安装PyTorch：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 1.3 配置环境变量

- [ ] 创建/检查 `.env` 文件
- [ ] 配置 `MLFLOW_TRACKING_URI=http://localhost:5001`
- [ ] 配置 `PACK_PROJECT_DIR=/root/pack`

***

## 阶段二：数据集准备

### 2.1 检查数据集目录结构

- [ ] 确认 `datasets/coco_style_fourclass/yolo_dataset_seg/` 存在
- [ ] 检查 `images/train/`、`images/test/` 目录
- [ ] 检查 `labels/train/`、`labels/test/` 目录
- [ ] 检查 `dataset.yaml` 配置文件

### 2.2 验证数据集完整性

- [ ] 训练集：6077张图片
- [ ] 验证集：655张图片
- [ ] 测试集：1000张图片
- [ ] 确认4个类别定义正确

***

## 阶段三：YOLO-cheap模型准备

### 3.1 获取模型配置

- [x] 下载 `yolov8-seg-cheap.yaml` 模型文件
- [x] 放置在项目根目录或 YOLO/ 目录

### 3.2 确认模型规格

- [ ] 网络层数：360层
- [ ] 参数量：2,657,674
- [ ] GFLOPs：10.4

***

## 阶段四：配置文件修改

### 4.1 修改训练配置文件

- [ ] 修改 `configs/hyp_lscd.yaml`：使用相对路径
- [ ] 修改 `configs/hyp_lscd_boundary.yaml`：使用相对路径
- [ ] 修改 `configs/pack_val_config.yaml`：使用相对路径
- [ ] 修改 `configs/pack_predict_config.yaml`：使用相对路径

### 4.2 确认数据集配置

- [ ] 检查 `datasets/coco_style_fourclass/yolo_dataset_seg/dataset.yaml`
- [ ] 确认路径配置正确
- [ ] 确认类别名称正确

***

## 阶段五：边界感知损失集成

### 5.1 备份原始损失文件

- [ ] 备份 ultralytics 的 loss.py

### 5.2 替换为边界感知损失

- [ ] 复制 `loss_with_boundary_fast.py` 到 ultralytics utils 目录

### 5.3 确认参数配置

- [ ] boundary: 1.5（边界像素权重倍数）
- [ ] boundary\_width: 3（边界宽度）

***

## 阶段六：训练流程

### 6.1 启动MLflow服务

- [ ] 在后台启动MLflow服务：`mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns`
- [ ] 验证MLflow服务可正常访问

### 6.2 基线训练

- [ ] 进入YOLO目录：`cd YOLO`
- [ ] 执行训练：`python pack_train.py --config ../configs/hyp_lscd.yaml --val`
- [ ] 监控训练过程和MLflow记录

### 6.3 边界感知损失训练

- [ ] 确保loss.py已替换为边界感知版本
- [ ] 执行训练：`python pack_train.py --config ../configs/hyp_lscd_boundary.yaml --val`
- [ ] 记录实验结果

### 6.4 遮挡增强训练

- [ ] 执行训练：`python pack_train_occlusion_aug.py --config ../configs/hyp_lscd.yaml --occlusion-aug --aug-ratio 0.3 --val`
- [ ] 记录实验结果

### 6.5 组合训练（边界感知 + 遮挡增强）

- [ ] 确保loss.py已替换为边界感知版本
- [ ] 执行训练：`python pack_train_occlusion_aug.py --config ../configs/hyp_lscd_boundary.yaml --occlusion-aug --aug-ratio 0.3 --val`
- [ ] 记录实验结果

***

## 阶段七：验证与推理

### 7.1 模型验证

- [ ] 执行验证：`python pack_val.py --config ../configs/pack_val_config.yaml`
- [ ] 检查mAP指标

### 7.2 模型推理

- [ ] 执行推理：`python pack_predict.py --config ../configs/pack_predict_config.yaml`
- [ ] 验证推理结果

***

## 阶段八：模型部署

### 8.1 保存训练模型

- [ ] 复制最佳权重到 `models/best.pt`

### 8.2 部署验证

- [ ] 确认模型可正常加载
- [ ] 确认Web服务可调用模型

***

## 预期结果对比

| 实验         | 整体Box | 整体Mask | inner-all | inner-occ | outer-all | outer-occ |
| ---------- | ----- | ------ | --------- | --------- | --------- | --------- |
| 基线         | 75.3% | -      | 82.4%     | 73.8%     | 86.2%     | 58.9%     |
| +边界感知      | 76.0% | 76.0%  | 82.0%     | 75.1%     | 86.8%     | 60.2%     |
| +边界感知+遮挡增强 | 76.0% | 76.0%  | 81.8%     | 75.6%     | 86.6%     | 60.2%     |

***

## 关键检查点

### 训练前检查

- [ ] 数据集已放置在 datasets/ 目录
- [ ] 依赖已完整安装
- [ ] .env 已正确配置
- [ ] MLflow 服务可正常访问
- [ ] loss.py 已备份

### 训练后验证

- [ ] 模型能正常加载训练
- [ ] MLflow 记录正常
- [ ] 训练结果符合预期
- [ ] 模型已保存到 models/

***

## 时间预估

| 阶段     | 预估时间  |
| ------ | ----- |
| 环境准备   | 30分钟  |
| 数据集准备  | 10分钟  |
| 模型配置   | 10分钟  |
| 基线训练   | 2-4小时 |
| 边界感知训练 | 2-4小时 |
| 遮挡增强训练 | 2-4小时 |
| 组合训练   | 2-4小时 |
| 验证与部署  | 30分钟  |

***

*规划创建时间：2026-05-04*
