# Pack 地堆检测系统 - 箱货检测与SKU匹配

基于深度学习的地堆箱货识别和SKU匹配项目，使用YOLOv8进行目标检测，ViT-S16 DINO进行特征匹配。

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Web 前端 (Vue 3)                        │
│              http://localhost:5173 (开发)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP REST API
┌─────────────────────────────▼───────────────────────────────┐
│                    Web 后端 (FastAPI)                       │
│              http://localhost:8000                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  BoxDetector │  │  SKUMatcher   │  │  可视化模块   │    │
│  │   (YOLOv8)   │  │  (DINO+OML)  │  │             │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 目录结构

```
pack/
├── core/                        # 核心算法模块（独立，可复用）
│   ├── detector/               # YOLO检测封装
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   ├── matcher/                # SKU匹配算法
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   └── sku_matcher.py
│   ├── utils/                  # 通用工具
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   ├── logger.py
│   │   └── pytorch_utils.py
│   └── visualizer.py           # 结果可视化
│
├── web/                        # Web应用
│   ├── backend/                # FastAPI后端
│   │   ├── api/                # API路由
│   │   │   ├── __init__.py
│   │   │   ├── task.py
│   │   │   ├── sku.py
│   │   │   ├── sku_review.py
│   │   │   ├── build.py
│   │   │   └── logs.py
│   │   ├── services/           # 业务服务层
│   │   │   ├── detect_match_service.py
│   │   │   ├── detection_service.py
│   │   │   └── match_service.py
│   │   ├── repositories/       # 数据访问层
│   │   │   └── task_repository.py
│   │   ├── models/             # SQLAlchemy模型
│   │   ├── schemas/            # Pydantic数据模型
│   │   ├── database.py         # 数据库配置
│   │   ├── main.py             # API入口
│   │   └── static/             # 前端静态资源
│   └── frontend/               # Vue.js前端
│       ├── src/
│       │   ├── api/            # API接口封装
│       │   │   ├── client.js
│       │   │   ├── taskApi.js
│       │   │   └── skuApi.js
│       │   ├── components/     # 组件库
│       │   ├── stores/         # Pinia状态管理
│       │   ├── utils/          # 工具函数
│       │   ├── App.vue
│       │   ├── main.js
│       │   └── style.css
│       ├── index.html
│       ├── package.json
│       └── vite.config.js
│
├── training/                   # 模型训练模块
│   ├── sku/                    # SKU特征库构建
│   │   ├── README.md
│   │   ├── build_sku_library.py
│   │   ├── extract_features.py
│   │   └── train_oml.py
│   └── yolo/                   # YOLO训练
│       ├── configs/
│       ├── runs/               # 训练输出
│       ├── train.py
│       └── predict.py
│
├── scripts/                    # 工具脚本
│   ├── data/                   # 数据处理
│   ├── detect/                 # 检测脚本
│   └── visualize/              # 可视化脚本
│
├── docs/                       # 文档
│   ├── 系统设计文档.md          # 系统架构设计
│   └── api-docs.md             # API接口文档
│
├── config.py                   # 全局配置
├── requirements.txt            # Python依赖
├── start-all.bat               # Windows一键启动
├── start-backend.bat           # 启动后端
├── start-frontend.bat          # 启动前端
└── start.sh                    # Linux启动脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- PyTorch 2.0+ (CUDA可选)

### 1. 安装依赖

```bash
# Python后端依赖
pip install -r requirements.txt

# 前端依赖
cd web/frontend
npm install
```

### 2. 启动服务

**方式一：一键启动（推荐）**

```bash
# Windows
start-all.bat

# Linux/Mac
bash start.sh
```

**方式二：手动启动**

```bash
# 启动后端（终端1）
cd web/backend
python main.py

# 启动前端（终端2）
cd web/frontend
npm run dev
```

### 3. 访问应用

- 前端地址：http://localhost:5173
- 后端API：http://localhost:8000
- API文档：http://localhost:8000/docs

## 📋 功能模块

| 模块 | 功能 | 说明 |
|------|------|------|
| 图片上传 | 支持单图/批量上传 | 拖拽或点击上传 |
| 目标检测 | YOLOv8箱货检测 | 自动识别图片中的箱体 |
| SKU匹配 | DINO特征匹配 | 每个箱体匹配Top-5候选 |
| 任务管理 | 任务列表、详情、删除 | 完整的任务生命周期管理 |
| SKU管理 | SKU库管理、图片上传 | 支持批量导入导出 |
| SKU审核 | 图片分配到SKU | 辅助标注工具 |

## 🛠️ 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 前端框架 | Vue | 3.4+ |
| 前端构建 | Vite | 5.0+ |
| 状态管理 | Pinia | 3.0+ |
| UI组件 | Element Plus | 2.4+ |
| 后端框架 | FastAPI | 0.100+ |
| 数据库 | SQLite | 内置 |
| 目标检测 | YOLOv8 | 最新 |
| 特征提取 | DINO ViT-S16 | 预训练 |
| 度量学习 | OML | 最新 |

## 📝 论文技术点

1. **YOLO-cheap 轻量化检测** - 使用YOLOv8n小型模型，平衡速度与精度
2. **拉普拉斯边界感知损失** - 在YOLO训练中加入边界感知，提升检测精度
3. **OML度量学习微调** - 使用OML库进行SKU特征库的度量学习训练
4. **DINO自监督学习** - 利用DINO预训练模型提取高质量特征

## 📄 文档

- `docs/系统设计文档.md` - 系统架构、设计决策、核心流程
- `docs/api-docs.md` - REST API接口规范
- `web/frontend/src/components/COMPONENTS.md` - 前端组件文档

## 📧 联系方式

如有问题，请联系开发人员。