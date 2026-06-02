# 仓储物品识别系统 - 箱货检测与SKU匹配

基于深度学习的地堆箱货识别和SKU匹配项目，使用YOLOv8进行目标检测，ViT-S16 DINO进行特征匹配。

## 📋 更新日志

### 2026-06-02
- ✨ **SKU硬删除**：SKU删除功能改为物理删除（数据库+图片+CSV+特征向量）
- 🗑️ **箱体删除**：新增独立API支持真实物理删除（数据库+裁剪图片+匹配结果）
- 📊 **分页功能**：任务列表和SKU列表均支持分页显示
- 🎨 **UI优化**：待识别标签改为蓝色，待审核保持黄色；删除多余按钮
- ⚙️ **Bug修复**：修复审核提交索引错位问题
- 🔄 **模型选择**：首页状态栏模型名称改为下拉选择显示（纯前端展示，无实际切换功能）

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
│   ├── visualizer.py           # 结果可视化
│   └── __init__.py
│
├── web/                        # Web应用
│   ├── backend/                # FastAPI后端
│   │   ├── api/                # API路由
│   │   │   ├── __init__.py
│   │   │   ├── task.py
│   │   │   ├── sku.py
│   │   │   ├── sku_review.py
│   │   │   ├── build.py
│   │   │   ├── logs.py
│   │   │   ├── review.py
│   │   │   ├── match.py
│   │   │   ├── detection.py
│   │   │   └── export.py
│   │   ├── services/           # 业务服务层
│   │   │   ├── detect_match_service.py
│   │   │   ├── detection_service.py
│   │   │   └── match_service.py
│   │   ├── repositories/       # 数据访问层
│   │   │   └── task_repository.py
│   │   ├── models/             # SQLAlchemy模型
│   │   │   ├── __init__.py
│   │   │   ├── task.py
│   │   │   ├── detection_box.py
│   │   │   ├── match_result.py
│   │   │   ├── sku.py
│   │   │   └── operation_log.py
│   │   ├── schemas/            # Pydantic数据模型
│   │   │   ├── __init__.py
│   │   │   └── schemas.py
│   │   ├── migrations/        # 数据库迁移
│   │   │   └── v1_to_v2.py
│   │   ├── database.py         # 数据库配置
│   │   ├── main.py             # API入口
│   │   ├── create_test_image.py
│   │   └── static/             # 前端静态资源
│   └── frontend/               # Vue.js前端
│       ├── src/
│       │   ├── api/            # API接口封装
│       │   │   ├── client.js
│       │   │   ├── taskApi.js
│       │   │   └── skuApi.js
│       │   ├── components/     # 组件库
│       │   │   ├── home/
│       │   │   │   ├── RecentTasks.vue
│       │   │   │   └── SysStatusBar.vue
│       │   │   ├── layout/
│       │   │   │   ├── PageHeader.vue
│       │   │   │   └── PageContainer.vue
│       │   │   ├── pages/
│       │   │   │   ├── HomePage.vue
│       │   │   │   ├── TaskListPage.vue
│       │   │   │   ├── SkuListPage.vue
│       │   │   │   └── SkuReviewPage.vue
│       │   │   ├── task/
│       │   │   │   ├── TaskStats.vue
│       │   │   │   ├── DetectionList.vue
│       │   │   │   ├── MatchResult.vue
│       │   │   │   ├── TaskNav.vue
│       │   │   │   ├── BatchResultCard.vue
│       │   │   │   └── BoxMatchDialog.vue
│       │   │   ├── sku/
│       │   │   │   └── SkuImage.vue
│       │   │   ├── upload/
│       │   │   │   ├── UploadArea.vue
│       │   │   │   └── FileList.vue
│       │   │   ├── ui/
│       │   │   │   ├── ActionMenu.vue
│       │   │   │   ├── FilterBar.vue
│       │   │   │   ├── FilterDropdown.vue
│       │   │   │   ├── TimeFilterDropdown.vue
│       │   │   │   ├── ImageViewer.vue
│       │   │   │   └── StatusBanner.vue
│       │   │   └── COMPONENTS.md
│       │   ├── stores/         # Pinia状态管理
│       │   │   └── app.js
│       │   ├── utils/          # 工具函数
│       │   │   └── taskUtils.js
│       │   ├── App.vue
│       │   ├── main.js
│       │   └── style.css
│       ├── index.html
│       ├── package.json
│       ├── package-lock.json
│       └── vite.config.js
│
├── training/                   # 模型训练模块
│   ├── sku/                    # SKU特征库构建
│   │   ├── README.md
│   │   ├── build_sku_library.py
│   │   ├── extract_features.py
│   │   ├── train_oml.py
│   │   ├── evaluate.py
│   │   ├── split_sku.py
│   │   └── test_matcher.py
│   └── yolo/                   # YOLO训练
│       ├── configs/
│       │   ├── hyp_lscd.yaml
│       │   ├── hyp_lscd_boundary.yaml
│       │   ├── pack_predict_config.yaml
│       │   └── pack_val_config.yaml
│       ├── runs/               # 训练输出
│       ├── train.py
│       ├── predict.py
│       ├── requirements.txt
│       └── loss_with_boundary_fast.py
│
├── scripts/                    # 工具脚本
│   ├── README.md
│   ├── data/                   # 数据处理
│   │   ├── coco2yolo_seg.py
│   │   └── filter_images.py
│   ├── detect/                 # 检测脚本
│   │   └── batch_detect_crop.py
│   └── visualize/              # 可视化脚本
│       └── visualize_annotations.py
│
├── docs/                       # 文档
│   └── 系统设计文档.md          # 系统架构设计
│
├── config.py                   # 全局配置
├── requirements.txt            # Python依赖
├── start-all.bat               # Windows一键启动
├── start-backend.bat           # 启动后端
├── start-frontend.bat          # 启动前端
├── stop-all.bat                # 停止所有服务
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

| 模块     | 功能                 | 说明                                 |
| -------- | -------------------- | ------------------------------------ |
| 图片上传 | 支持单图/批量上传    | 拖拽或点击上传                       |
| 目标检测 | YOLOv8箱货检测       | 自动识别图片中的箱体                 |
| SKU匹配  | DINO特征匹配         | 每个箱体匹配Top-5候选                |
| 任务管理 | 任务列表、详情、删除 | 完整的任务生命周期管理，支持分页显示 |
| SKU管理  | SKU库管理、图片上传  | 支持批量导入导出、物理删除           |
| 箱体管理 | 箱体删除             | 支持单独删除任务中的单个箱体         |
| SKU审核  | 图片分配到SKU        | 辅助标注工具，审核结果可回溯         |

## 🛠️ 技术栈

| 层级     | 技术         | 版本   |
| -------- | ------------ | ------ |
| 前端框架 | Vue          | 3.4+   |
| 前端构建 | Vite         | 5.0+   |
| 状态管理 | Pinia        | 3.0+   |
| UI组件   | Element Plus | 2.4+   |
| 后端框架 | FastAPI      | 0.100+ |
| 数据库   | SQLite       | 内置   |
| 目标检测 | YOLOv8       | 最新   |
| 特征提取 | DINO ViT-S16 | 预训练 |
| 度量学习 | OML          | 最新   |

## 📝 论文技术点

1. **YOLO-cheap 轻量化检测** - 使用YOLOv8n小型模型，平衡速度与精度
2. **拉普拉斯边界感知损失** - 在YOLO训练中加入边界感知，提升检测精度
3. **OML度量学习微调** - 使用OML库进行SKU特征库的度量学习训练
4. **DINO自监督学习** - 利用DINO预训练模型提取高质量特征

## 📄 文档

- `docs/系统设计文档.md` - 系统架构、设计决策、核心流程
- `web/frontend/src/components/COMPONENTS.md` - 前端组件文档

## 📧 联系方式

如有问题，请联系开发人员。