# 系统架构设计

## 1. 项目概述

本项目是一个基于深度学习的箱货检测与SKU匹配系统，主要功能包括：
- 使用YOLOv8进行箱体目标检测
- 基于ViT-S16 DINO模型的SKU特征匹配
- 提供Web界面展示检测和匹配结果

## 2. 技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 前端 (Vue.js)                      │
│              负责用户交互、结果展示、图片上传                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Web 后端 (FastAPI)                        │
│              提供RESTful API、路由转发、静态资源              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core 核心模块                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Detector   │  │   Matcher   │  │    Utils    │          │
│  │  (YOLO)     │  │   (SKU)     │  │ (图像处理)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
pack/
├── core/                      # 核心算法模块
│   ├── detector/              # YOLO检测封装
│   │   └── yolo_detector.py
│   ├── matcher/               # SKU匹配算法
│   │   └── sku_matcher.py
│   ├── utils/                 # 通用工具
│   │   ├── logger.py
│   │   ├── image_utils.py
│   │   └── pytorch_utils.py
│   └── visualizer.py          # 结果可视化
├── data/                     # 数据和模型
│   ├── models/                # 模型权重
│   │   ├── yolo/best.pt
│   │   └── sku/vits16_dino.pth
│   ├── sku_library/           # SKU特征库
│   └── .yolo/, .ultralytics/  # 运行时配置
├── SKU/                       # SKU训练模块（独立）
├── YOLO/                      # YOLO训练模块（独立）
├── web/                       # Web应用
│   ├── backend/               # FastAPI后端
│   │   ├── schemas/           # Pydantic数据模型
│   │   ├── static/            # 前端静态资源
│   │   ├── config.py
│   │   └── main.py
│   └── frontend/              # Vue.js前端
├── scripts/                   # 通用工具脚本
└── docs/                      # 文档
```

## 3. 核心模块设计

### 3.1 BoxDetector（检测模块）

**职责**：封装YOLOv8模型，提供目标检测功能

**主要接口**：
- `detect_single_image()`: 对单张图片进行检测
- `is_ready()`: 检查模型是否加载成功

**核心流程**：
1. 加载YOLOv8预训练模型
2. 对输入图片进行推理
3. 返回检测框坐标、置信度、类别信息

### 3.2 SKUMatcher（匹配模块）

**职责**：实现基于特征向量的SKU匹配

**主要接口**：
- `extract_feature()`: 提取图片特征向量
- `match_sku()`: 执行相似度匹配

**匹配算法**：
1. 使用ViT-S16 DINO提取图片特征（384维）
2. 计算查询特征与库中特征的余弦相似度
3. 应用Ratio Test过滤误匹配
4. 返回Top-5匹配结果

### 3.3 图像处理工具（image_utils）

**功能**：
- `crop_box()`: 根据检测框裁剪图片
- `resize_with_padding()`: 等比例缩放并填充
- `filter_small_boxes()`: 过滤过小的检测框
- `image_to_base64()` / `base64_to_image()`: 图片编解码

## 4. Web层设计

### 4.1 后端架构

使用FastAPI构建RESTful API，主要接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 系统健康检查 |
| `/api/detect` | POST | 仅检测 |
| `/api/match` | POST | 仅匹配 |
| `/api/detect-and-match` | POST | 检测+匹配 |
| `/api/skus` | GET | SKU列表查询 |
| `/api/sku-image/{id}/{name}` | GET | SKU图片获取 |

### 4.2 前端架构

使用Vue.js + Pinia构建单页应用：
- **App.vue**: 主组件，协调各模块
- **stores/app.js**: 状态管理
- **api/detector.js**: API调用封装

## 5. 数据流

### 5.1 检测+匹配流程

```
用户上传图片
    ↓
前端POST /api/detect-and-match
    ↓
后端调用BoxDetector.detect_single_image()
    ↓
过滤小框，裁剪检测区域
    ↓
对每个检测框：
    1. 调用SKUMatcher.extract_feature()提取特征
    2. 调用SKUMatcher.match_sku()进行匹配
    ↓
组装结果，返回前端
    ↓
前端展示检测结果和Top-5匹配候选
```

## 6. 配置管理

采用集中式配置管理（`config.py`）：
- 模型路径配置
- 阈值参数配置（置信度、匹配阈值等）
- 路径配置（数据目录、模型目录等）

## 7. 扩展性设计

### 7.1 模块独立性
- 核心算法（core/）与Web层完全解耦
- 可单独导入使用，如：`from core.detector import BoxDetector`

### 7.2 配置可扩展
- 所有阈值、路径均可通过config.py配置
- 支持环境变量覆盖

### 7.3 训练流程独立
- SKU和YOLO训练作为独立模块
- 不影响Web服务的运行
