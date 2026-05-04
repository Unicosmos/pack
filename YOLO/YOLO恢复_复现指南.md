# YOLO训练项目复现指南

本文档记录了YOLO纸箱分割项目的完整恢复过程，包含所有配置文件、训练命令和实验结果。

---

## 1. 项目目录结构

```
<项目根目录>/
├── YOLO/                        # YOLO训练模块
│   ├── pack_train.py            # YOLO训练脚本
│   ├── pack_train_occlusion_aug.py  # 遮挡增强版训练
│   ├── pack_predict.py          # YOLO推理脚本
│   ├── pack_val.py              # YOLO验证脚本
│   └── register_modules.py      # 自定义模块注册
├── SKU/                         # SKU匹配模块
│   ├── sku_augmentation.py    # SKU图片增强
│   ├── sku_matcher.py          # SKU匹配器
│   ├── sku_review.py          # SKU审核（Gradio）
│   └── smart_sku_manager.py   # SKU管理工具
│   └── feature_extractor.py   # ViT-S16特征提取器
├── utils/                       # 工具模块
│   ├── box_detector.py         # 纸箱检测器
│   ├── split_box.py            # 箱体分割工具
│   ├── coco2yolo_seg.py       # COCO格式转换
│   ├── occlusion_aug_fast.py   # 快速遮挡增强
│   └── find_augmented_samples.py  # 查找增强样本
├── configs/                     # 配置文件
│   ├── hyp_lscd.yaml           # 核心训练配置
│   ├── hyp_lscd_boundary.yaml  # 边界感知损失训练配置
│   ├── hyp_lscd_no_aug.yaml   # 关闭增强版训练配置
│   ├── pack_predict_config.yaml  # 预测配置
│   └── pack_val_config.yaml    # 验证配置
├── models/                      # 模型权重
│   └── best.pt                  # 训练后的模型
├── datasets/                    # 数据集存放目录
│   └── coco_style_fourclass/
│       └── yolo_dataset_seg/
│           ├── dataset.yaml   # 数据集配置
│           ├── images/
│           │   ├── train/
│           │   └── test/
│           └── labels/
│               ├── train/
│               └── test/
├── augmented_datasets/          # 增强数据集（训练时自动生成）
├── models/                    # 训练输出目录
├── .env                       # 环境变量配置（新建）
└── loss_with_boundary_fast.py  # 边界感知损失（新建）
```

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# 进入项目根目录
cd <项目根目录>

# 安装核心依赖
pip install ultralytics scipy opencv-python tqdm pyyaml python-dotenv

# 安装MLflow（用于训练记录）
pip install mlflow

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.2 配置环境变量

新建 `.env` 文件：

```bash
# .env 文件内容
MLFLOW_TRACKING_URI=http://localhost:5001
PACK_PROJECT_DIR=<项目根目录>
```

### 2.3 数据集准备

```bash
# 将数据集放置在 datasets/ 目录下
# 目录结构见 1. 项目目录结构
```

---

## 3. YOLO-cheap复现修改点

### 3.1 模型定义
- 模型文件：`yolov8-seg-cheap.yaml`
- 来源：YOLO-cheap论文GitHub仓库
- 下载链接：https://github.com/AlanDoan/YOLO-Cheap

下载后放置在项目根目录或YOLO/目录

### 3.2 模型规格
| 指标 | 数值 |
|------|------|
| 网络层数 | 360层 |
| 参数量 | 2,657,674 |
| GFLOPs | 10.4 |

### 3.3 核心模块
```
CheapConv - 分解卷积（水平+垂直一维卷积）
├── Depthwise 1D H（水平）
└── Depthwise 1D V（垂直）
    └── Pointwise 1x1

RepDS - 重参数化模块
RepHead - 检测头重参数化
```

### 3.4 数据集信息
- 数据集名称：SCD（Smart Carton Dataset）
- 总样本数：7732张
- 类别数：4类
- 划分：
  - 训练集：6077张
  - 验证集：655张
  - 测试集：1000张

### 3.5 类别定义
| 类别ID | 类别名称 |
|--------|----------|
| 0 | Carton-inner-all |
| 1 | Carton-inner-occlusion |
| 2 | Carton-outer-all |
| 3 | Carton-outer-occlusion |

---

## 4. 配置文件说明

### 4.1 修改配置文件路径

修改 `configs/hyp_lscd.yaml`：

```yaml
# 修改前
mlflow-uri: http://localhost:5001
project: /root/source/data2/hyg/projects/pack/runs/segment
data: /root/source/data2/hyg/projects/pack/coco_style_fourclass/yolo_dataset_seg/dataset.yaml

# 修改后（使用相对路径）
mlflow-uri: http://localhost:5001
project: ./runs/segment
data: ./datasets/coco_style_fourclass/yolo_dataset_seg/dataset.yaml
```

其他配置文件同理修改。

### 4.2 数据集配置 dataset.yaml

```yaml
path: ./datasets/coco_style_fourclass/yolo_dataset_seg
train: images/train
val: images/val
test: images/test

names:
  0: Carton-inner-all
  1: Carton-inner-occlusion
  2: Carton-outer-all
  3: Carton-outer-occlusion
```

---

## 5. 边界感知损失集成方式

### 5.1 原理
在BCE损失后增加边界权重图：
1. 使用拉普拉斯算子提取mask边界
2. max_pool膨胀边界区域
3. 边界像素权重 × boundary倍数

### 5.2 两个版本对比
| 版本 | 文件 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| scipy版 | `loss_with_boundary.py` | 慢30-50% | 更精确 | 离线评估 |
| PyTorch版 | `loss_with_boundary_fast.py` | 慢5-10% | 足够 | **训练使用** |

### 5.3 集成步骤

**Step 1**: 备份原始loss.py
```bash
cp $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py \
   $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py.bak
```

**Step 2**: 替换为边界感知损失
```bash
# 确保在项目根目录
cd <项目根目录>
cp loss_with_boundary_fast.py \
   $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py
```

**Step 3**: 训练时使用对应配置
```bash
cd YOLO
python pack_train.py --config ../configs/hyp_lscd_boundary.yaml --val
```

### 5.4 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| boundary | 1.5 | 边界像素权重倍数（1.0=关闭） |
| boundary_width | 3 | 边界宽度（像素） |

---

## 6. 遮挡增强使用方法

### 6.1 独立预处理方式
```bash
cd <项目根目录>
python utils/occlusion_aug_fast.py \
    --source datasets/coco_style_fourclass/yolo_dataset_seg/dataset.yaml \
    --output ./augmented_datasets \
    --aug-ratio 0.3
```

### 6.2 集成训练方式（推荐）
```bash
cd YOLO
python pack_train_occlusion_aug.py \
    --config ../configs/hyp_lscd.yaml \
    --occlusion-aug \
    --aug-ratio 0.3 \
    --val
```

### 6.3 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| --occlusion-aug | False | 启用遮挡增强 |
| --aug-ratio | 0.3 | 30%样本进行增强 |
| --aug-prob | 0.5 | 每个样本50%概率增强 |

---

## 7. MLflow使用方法

### 7.1 启动MLflow服务

```bash
# 在项目根目录下
cd <项目根目录>

# 启动MLflow服务（后台运行）
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns

# 浏览器访问：http://<云主机IP:5001
```

### 7.2 MLflow记录内容
- 训练参数
- 训练曲线
- 模型权重
- 实验结果

---

## 8. 云主机完整训练流程

### 8.1 进入项目根目录
```bash
cd <项目根目录>
```

### 8.2 检查依赖
```bash
# 确认已安装依赖
pip list | grep -E "ultralytics|mlflow|torch|scipy|opencv|python-dotenv

# 未安装则执行 2.1 步骤
```

### 8.3 基线训练
```bash
# 启动MLflow服务
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns &

# 训练
cd YOLO
python pack_train.py --config ../configs/hyp_lscd.yaml --val
```

### 8.4 边界感知损失训练
```bash
# Step 1: 替换loss.py
cd <项目根目录>
cp loss_with_boundary_fast.py \
   $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py

# Step 2: 训练
cd YOLO
python pack_train.py --config ../configs/hyp_lscd_boundary.yaml --val
```

### 8.5 遮挡增强训练
```bash
cd YOLO
python pack_train_occlusion_aug.py \
    --config ../configs/hyp_lscd.yaml \
    --occlusion-aug \
    --aug-ratio 0.3 \
    --val
```

### 8.6 组合训练（边界感知 + 遮挡增强）
```bash
# Step 1: 替换loss.py
cd <项目根目录>
cp loss_with_boundary_fast.py \
   $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py

# Step 2: 训练
cd YOLO
python pack_train_occlusion_aug.py \
    --config ../configs/hyp_lscd_boundary.yaml \
    --occlusion-aug \
    --aug-ratio 0.3 \
    --val
```

---

## 9. 验证与推理

### 9.1 模型验证
```bash
cd YOLO
python pack_val.py --config ../configs/pack_val_config.yaml
```

### 9.2 模型推理
```bash
cd YOLO
python pack_predict.py --config ../configs/pack_predict_config.yaml
```

---

## 10. 模型部署

### 10.1 复制训练好的模型到Web系统

```bash
# 训练后复制到 models/
cp YOLO/runs/segment/xxxx/weights/best.pt models/

# 再复制到 web/backend/models/
mkdir -p web/backend/models
cp models/best.pt web/backend/models/
```

---

## 11. 实验结果数据

### 11.1 消融实验结果（mAP@50:95）
| 实验 | 整体Box | 整体Mask | inner-all | inner-occ | outer-all | outer-occ |
|------|---------|----------|-----------|-----------|-----------|-----------|
| 基线 | 75.3% | - | 82.4% | 73.8% | 86.2% | 58.9% |
| +边界感知 | 76.0% | 76.0% | 82.0% | **75.1%(+1.3%) | 86.8% | **60.2%(+1.3%) |
| +边界感知+遮挡增强 | 76.0% | 76.0% | 81.8% | **75.6%** | 86.6% | **60.2%** |

### 11.2 关键发现
1. 边界感知损失对occlusion类别提升显著（+1.3%）
2. 遮挡增强对inner-occlusion类别有额外提升
3. outer-occlusion类别性能最低，是主要瓶颈

---

## 12. 附录：快速操作清单

### 12.1 首次部署前检查
- [ ] 数据集已放置在 datasets/ 目录
- [ ] 依赖已完整安装
- [ ] .env 已正确配置
- [ ] MLflow 服务可正常访问
- [ ] loss.py 已备份

### 12.2 部署后验证
- [ ] 模型能正常加载训练
- [ ] MLflow 记录正常
- [ ] 训练结果符合预期
- [ ] 模型已保存到 models/

---

*文档更新时间：2026-05-04*

