# Pack 项目 - 箱货检测与SKU匹配系统

基于深度学习的地堆箱货识别和SKU匹配项目，使用YOLOv8进行目标检测，ViT-S16 DINO进行特征匹配。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 前端 (Vue.js)                      │
│              http://localhost:5173 (开发)                    │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP API
┌─────────────────────────────▼───────────────────────────────┐
│                    Web 后端 (FastAPI)                       │
│              http://localhost:8000                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  BoxDetector │  │  SKUMatcher   │  │  可视化模块   │    │
│  │   (YOLO)     │  │    (DINO)    │  │             │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
pack/
├── core/                       # 核心算法模块（独立，可复用）
│   ├── detector/              # YOLO检测封装
│   │   └── yolo_detector.py
│   ├── matcher/               # SKU匹配算法
│   │   └── sku_matcher.py
│   ├── utils/                 # 通用工具
│   │   ├── logger.py
│   │   ├── image_utils.py
│   │   └── pytorch_utils.py   # PyTorch兼容性
│   └── visualizer.py          # 结果可视化
│
├── data/                      # 数据和模型
│   ├── models/                # 模型权重
│   │   ├── yolo/best.pt
│   │   └── sku/vits16_dino.pth
│   ├── sku_library/           # SKU特征库
│   │   ├── sku_features.npy   # 特征矩阵 (197, 384)
│   │   ├── sku_library.csv    # 图片索引
│   │   └── images/            # SKU图片
│   └── .yolo/, .ultralytics/ # 运行时配置
│
├── SKU/                       # SKU训练模块（独立）
│   ├── sku_augmentation.py    # 图片增强
│   ├── build_library.py       # 建库脚本
│   ├── feature_extractor.py   # 特征提取
│   └── sku_review.py          # Gradio标注工具
│
├── YOLO/                      # YOLO训练模块（独立）
│   ├── configs/               # 训练配置
│   └── runs/                  # 训练输出
│
├── web/                       # Web应用
│   ├── backend/               # FastAPI后端
│   │   ├── schemas/           # Pydantic数据模型
│   │   │   └── schemas.py
│   │   ├── static/            # 前端静态资源
│   │   ├── config.py          # 配置管理
│   │   └── main.py            # API入口
│   └── frontend/              # Vue.js前端
│       ├── src/               # 源代码
│       ├── package.json       # Node依赖
│       └── vite.config.js     # Vite配置
│
├── scripts/                   # 通用工具脚本
│   ├── filter_images.py       # 图片过滤
│   └── split_box.py           # 数据集划分
│
├── docs/                      # 文档
│   ├── ARCHITECTURE.md        # 架构设计
│   └── API.md                 # API接口文档
│
├── requirements.txt           # Python依赖
└── README.md
```

## 环境要求

### Python环境

- Python 3.10+
- CUDA 12.1+ (GPU) 或 CPU-only
- NVIDIA GPU (推荐RTX 4090用于训练)

### Node环境

- Node.js 18+
- npm

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装PyTorch (根据你的CUDA版本)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
# CPU版本: https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd web/frontend
npm install
```

### 3. 准备模型文件

```bash
# YOLO模型
mkdir -p data/models/yolo
cp YOLO/runs/xxx/weights/best.pt data/models/yolo/

# SKU特征模型（如有）
mkdir -p data/models/sku
```

### 4. 启动后端服务

```bash
cd web/backend
python main.py
# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

访问：
- API文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>

### 5. 启动前端服务（开发模式）

```bash
cd web/frontend
npm run dev
```

访问：<http://localhost:5173>

### 6. 构建前端（生产模式）

```bash
cd web/frontend
npm run build
# 构建结果在 web/backend/static/ 目录
```

## SKU建库流程

```bash
# Step 1: 切割源图片中的箱体
python scripts/box_detector.py \
    --images SKU/source \
    --output SKU/crops

# Step 2: Gradio人工审核标注
python SKU/sku_review.py

# Step 3: 图片增强
python SKU/sku_augmentation.py \
    --input SKU/sku_output \
    --output data/sku_library

# Step 4: 构建特征库
python SKU/build_library.py \
    --input data/sku_library \
    --output data/sku_library
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 系统健康检查 |
| `/api/detect` | POST | 仅检测 |
| `/api/detect-and-match` | POST | 检测+匹配 |
| `/api/match` | POST | 仅SKU匹配 |
| `/api/skus` | GET | SKU列表 |
| `/api/sku-image/{id}/{name}` | GET | SKU图片 |

详细API文档请参考 [docs/API.md](docs/API.md)

## 技术栈

### 后端

- FastAPI - Web框架
- Ultralytics YOLOv8 - 目标检测
- ViT-S16 DINO - 特征提取
- Pillow/OpenCV - 图像处理

### 前端

- Vue 3 - 前端框架
- Element Plus - UI组件库
- Pinia - 状态管理
- Vite - 构建工具

### 训练

- PyTorch 2.2.2 - 深度学习框架
- timm - 预训练模型库
- MLflow - 实验记录

## 项目状态

| 模块 | 状态 | 说明 |
|------|------|------|
| YOLO训练 | ✅ 完成 | YOLOv8实例分割 |
| 后端API | ✅ 完成 | FastAPI RESTful API |
| 前端界面 | ✅ 完成 | Vue.js单页应用 |
| SKU建库工具 | ✅ 完成 | 增强+特征提取 |
| 核心模块重构 | ✅ 完成 | 模块化架构 |

## 文档

- [系统架构设计](docs/ARCHITECTURE.md) - 详细架构文档
- [API接口文档](docs/API.md) - API使用指南

---

*最后更新：2026-05-12*
