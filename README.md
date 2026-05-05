# LSCD 项目 - 箱货识别与SKU匹配系统

基于LSCD数据集的地堆箱货识别和SKU匹配项目，使用YOLOv8-cheap进行实例分割，ViT-S16 DINO进行特征匹配。

***

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Vue.js)                         │
│              http://localhost:5173                       │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP API
┌─────────────────────▼───────────────────────────────────┐
│                   后端 (FastAPI)                        │
│              http://localhost:8000                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ YOLO检测器   │  │  SKU匹配器   │  │  可视化模块  │   │
│  │ (best.pt)    │  │ (DINO+CSV)  │  │             │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

***

## 目录结构

```
pack/
├── models/                      # 模型文件（需创建）
│   └── best.pt                  # YOLO分割模型
│
├── sku_library/                 # SKU特征库（需创建）
│   ├── sku_library.csv          # 图片索引
│   ├── sku_features.npy         # 特征矩阵
│   └── 000001/                  # SKU图片
│
├── web/                         # Web应用
│   ├── backend/                  # FastAPI后端
│   │   ├── main.py              # API入口
│   │   ├── config.py            # 配置管理
│   │   ├── core/                # 核心模块
│   │   │   ├── matcher.py       # SKU匹配器
│   │   │   └── visualizer.py    # 结果可视化
│   │   ├── models/schemas.py    # 数据模型
│   │   └── requirements.txt      # Python依赖
│   │
│   └── frontend/                # Vue.js前端
│       ├── src/                 # 源代码
│       ├── package.json         # Node依赖
│       └── vite.config.js       # Vite配置
│
├── YOLO/                        # YOLO训练模块
│   ├── configs/                  # 训练配置
│   ├── runs/                    # 训练输出
│   │   └── lscd_segmentation_xxx/
│   │       └── weights/best.pt  # 训练好的模型
│   └── loss_with_boundary_fast.py  # 边界感知损失
│
├── SKU/                         # SKU处理模块
│   ├── sku_augmentation.py      # 图片增强
│   ├── feature_extractor.py      # 特征提取
│   └── sku_review.py             # Gradio标注工具
│
└── utils/                       # 工具脚本
    ├── box_detector.py           # 箱体检测
    └── ...
```

***

## 环境要求

### Python环境

- Python 3.10+
- CUDA 11.8+ / CUDA 12.1+
- NVIDIA GPU (推荐RTX 4090)

### Node环境

- Node.js 18+
- npm

***

## 快速开始

### 1. 安装后端依赖

```bash
cd pack/web/backend
pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd pack/web/frontend
npm install
```

### 3. 准备模型文件

```bash
# 复制YOLO模型
mkdir -p pack/models
cp pack/YOLO/runs/lscd_segmentation_xxx/weights/best.pt pack/models/

```

### 4. 启动后端服务

```bash
cd pack/web/backend
conda activate pack  # 或你的虚拟环境
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

后端运行后访问：

- API文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>

### 5. 启动前端服务

````bash
先添加到 PATH（当前会话有效）

```
$env:Path = "C:\Program Files\nodejs;$env:Path"

```

cd pack/web/frontend
npm run dev
````

前端运行后访问：<http://localhost:5173>

***

## 系统使用流程

### 完整建库流程

```bash
# Step 1: 切割源图片中的箱体
python pack/utils/box_detector.py \
    --images pack/SKU/source \
    --model pack/models/best.pt \
    --output pack/SKU/crops

# Step 2: Gradio标注SKU（运行后访问 http://localhost:7860）
python pack/SKU/sku_review.py

# Step 3: 图片增强
python pack/SKU/sku_augmentation.py \
    --input pack/SKU/sku_output \
    --output pack/sku_library

# Step 4: 提取特征
python pack/SKU/feature_extractor.py \
    --input pack/sku_library \
    --output pack/sku_library/sku_features.npy
```

### Web使用流程

1. 打开前端页面 <http://localhost:5173>
2. 上传包含箱货的图片
3. 点击"开始检测"
4. 查看检测结果和SKU匹配信息

***

## API接口

| 接口                      | 方法   | 说明     |
| ----------------------- | ---- | ------ |
| `/api/health`           | GET  | 系统健康检查 |
| `/api/detect`           | POST | 仅检测    |
| `/api/detect-and-match` | POST | 检测+匹配  |
| `/api/match`            | POST | 仅SKU匹配 |
| `/api/skus`             | GET  | SKU列表  |

***

## 模型说明

| 模型       | 路径               | 说明                    |
| -------- | ---------------- | --------------------- |
| YOLO分割模型 | `models/best.pt` | YOLOv8-cheap，检测+分割箱体  |
| SKU特征库   | `sku_library/`   | ViT-S16 DINO提取的384维特征 |

***

## 训练YOLO模型

详细训练流程请参考 [YOLO/YOLO恢复\_复现指南.md](YOLO/YOLO恢复_复现指南.md)

```bash
cd pack/YOLO

# 基线训练
python pack_train.py --config configs/hyp_lscd.yaml --val

# 边界感知损失训练
cp loss_with_boundary_fast.py $(python -c "import ultralytics; print(ultralytics.__path__[0])")/utils/loss.py
python pack_train.py --config configs/hyp_lscd_boundary.yaml --val
```

***

## 技术栈

### 后端

- FastAPI - Web框架
- Ultralytics YOLO - 目标检测
- ViT-S16 DINO - 特征提取
- Pillow/OpenCV - 图像处理

### 前端

- Vue 3 - 前端框架
- Element Plus - UI组件库
- Pinia - 状态管理
- Axios - HTTP客户端
- Vite - 构建工具

### 训练

- PyTorch - 深度学习框架
- MLflow - 实验记录

***

## 项目状态

| 模块      | 状态   | 说明                                           |
| ------- | ---- | -------------------------------------------- |
| YOLO训练  | ✅ 完成 | 已在RTX 4090上训练                                |
| 后端API   | ✅ 完成 | FastAPI框架                                    |
| 前端界面    | ✅ 完成 | Vue.js界面                                     |
| SKU建库工具 | ✅ 完成 | 增强+特征提取                                      |
| YOLO模型  | ✅ 就绪 | `runs/lscd_segmentation_xxx/weights/best.pt` |
| SKU特征库  | ⏳ 待建 | 需要执行建库流程                                     |

***

## 常见问题

**Q: 前端无法访问？**
A: 确保后端和前端服务都已启动，检查端口是否被占用。

**Q: 检测结果为空？**
A: 检查YOLO模型是否正确放置在 `models/best.pt`。

**Q: SKU匹配全为unmatched？**
A: 确保 `sku_library/` 目录下有正确的 `sku_library.csv` 和 `sku_features.npy`。

***

*最后更新：2026-05-04*
