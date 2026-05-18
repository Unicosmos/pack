# Pack Web 应用

基于深度学习的箱货检测与SKU匹配系统。

## 系统功能

- ✅ **用户认证** - JWT Token 登录/注册
- ✅ **单图/批量检测** - 支持一次上传多张图片
- ✅ **SKU匹配** - 特征提取 + 相似度匹配
- ✅ **Top-5匹配结果** - 每个检测框展示5个候选
- ✅ **任务管理** - 列表、详情、删除、导出
- ✅ **SKU管理** - 列表、搜索、导出
- ✅ **参数配置** - 置信度/匹配阈值可视化调节
- ✅ **结果可视化** - 检测框、匹配图片直观展示

## 快速开始

### 1. 安装依赖

```bash
conda activate pack
cd d:\A_pack\pack
pip install -r requirements.txt
```

### 2. 启动后端

```bash
cd d:\A_pack\pack\web\backend
python main.py
```

后端服务运行在：**http://localhost:8000**

### 3. 访问应用

打开浏览器访问：`http://localhost:8000`

默认账号：`admin / admin123`

## 项目结构

```
pack/
├── web/
│   ├── backend/              # FastAPI后端
│   │   ├── api/              # API路由
│   │   │   ├── auth.py       # 用户认证
│   │   │   ├── sku.py        # SKU管理
│   │   │   └── task.py       # 任务管理
│   │   ├── models/           # 数据库模型
│   │   ├── schemas/          # Pydantic模型
│   │   ├── static/           # 前端静态资源
│   │   ├── config.py         # 配置文件
│   │   ├── database.py       # 数据库配置
│   │   ├── auth.py           # 认证核心
│   │   └── main.py           # 入口文件
│   └── frontend/             # Vue3前端源码
│       ├── src/
│       │   ├── components/   # 页面组件
│       │   ├── api/          # API客户端
│       │   └── stores/       # 状态管理
│       └── package.json
├── data/                     # 数据目录
│   ├── models/               # 模型权重
│   │   ├── yolo/            # YOLO模型
│   │   └── sku/             # SKU模型
│   ├── sku_library/          # SKU特征库
│   ├── uploads/              # 上传图片
│   └── pack.db               # SQLite数据库
├── SKU/                      # SKU训练模块
├── YOLO/                     # YOLO训练模块
├── core/                     # 核心算法模块
├── docs/                     # 文档
└── requirements.txt          # 依赖清单
```

## API接口

### 认证接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/me` | GET | 获取当前用户 |

### 检测接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 系统健康检查 |
| `/api/detect-and-match` | POST | 单图检测+匹配 |
| `/api/detect-batch` | POST | 批量检测（开发中） |

### 任务接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/tasks` | GET | 任务列表（分页） |
| `/api/tasks/{id}` | GET | 任务详情 |
| `/api/tasks/{id}` | DELETE | 删除任务 |
| `/api/tasks/upload` | POST | 上传图片 |
| `/api/tasks/upload-batch` | POST | 批量上传 |
| `/api/tasks/stats/summary` | GET | 任务统计 |
| `/api/tasks/export/csv` | GET | 导出CSV |

### SKU接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/skus` | GET | SKU列表（分页/搜索） |
| `/api/skus/{id}` | GET | SKU详情 |
| `/api/skus/stats` | GET | SKU统计 |
| `/api/skus/export/download` | GET | 导出CSV |
| `/api/skus/sync-from-csv` | POST | 从CSV同步 |

## 配置说明

配置文件位置：`web/backend/config.py`

```python
# 主要配置项
DATA_DIR          # 数据目录（data/）
MODEL_PATH       # YOLO模型路径
SKU_MODEL_PATH    # SKU模型路径
SKU_LIBRARY_DIR   # SKU库路径

# 阈值配置
CONF_THRESHOLD    # 置信度阈值（默认0.5）
MATCH_THRESHOLD   # 匹配阈值（默认0.85）
```

## 数据存储

| 类型 | 位置 |
|------|------|
| 用户数据 | `data/pack.db` (SQLite) |
| 上传图片 | `data/uploads/` |
| SKU库 | `data/sku_library/` |
| 模型权重 | `data/models/` |

## 技术栈

**后端**
- FastAPI - Web框架
- SQLAlchemy - ORM
- PyJWT - Token认证
- Ultralytics YOLO - 目标检测

**前端**
- Vue 3 - 渐进式框架
- Pinia - 状态管理
- Vite - 构建工具

**数据**
- SQLite - 轻量级数据库
- NumPy - 特征矩阵

## 开发指南

### 前端开发

```bash
cd web/frontend
npm install
npm run dev    # 开发模式
npm run build  # 生产构建
```

### 后端API文档

启动后访问：**http://localhost:8000/docs**（Swagger UI）

## 系统截图预览

1. **登录页** - 简洁的用户认证界面
2. **首页** - 上传图片、参数配置、检测结果展示
3. **任务列表** - 历史任务管理、分页筛选
4. **SKU管理** - SKU库浏览、搜索、导出

## 下一步

1. 把YOLO模型 `best.pt` 放入 `data/models/yolo/` 目录
2. 把SKU模型 `vits16_dino.pth` 放入 `data/models/sku/` 目录
3. 使用 `SKU/` 目录下工具准备SKU库
4. 访问 http://localhost:8000 测试完整流程

## 文档

- [系统架构文档](../docs/ARCHITECTURE.md)
- [API接口文档](../docs/API.md)
- [SKU建库流程](../SKU/README.md)