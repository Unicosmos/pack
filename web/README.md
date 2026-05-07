****# Pack Web 应用

## 快速开始

### 启动后端

```bash
conda activate pack
cd web/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 访问应用

打开浏览器访问：`http://localhost:8000`

## 项目结构

```
pack/
├── web/
│   ├── backend/
│   │   ├── main.py          # FastAPI后端（单文件）
│   │   ├── requirements.txt # Python依赖
│   │   └── static/          # 前端静态文件
│   │       └── index.html   # 单页应用
│   ├── frontend/            # 原Vue3版本（需要Node）
│   └── README.md
├── models/                  # 放置best.pt模型文件
├── sku_library/             # SKU库
└── ...                      # 其他原项目文件
```

## API接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/health` | GET | 健康检查 |
| `/api/detect-and-match` | POST | 检测+匹配 |
| `/api/skus` | GET | SKU列表 |
| `/docs` | GET | 自动API文档 |

## 功能

- 图片上传
- 箱体检测
- SKU匹配（带双重验证机制）
- Top-5 匹配结果展示
- 置信度/匹配度阈值调节
- 结果可视化

## 下一步

1. 把YOLO模型 `best.pt` 放入 `pack/models/` 目录
2. 使用 `SKU/` 目录下工具准备SKU库
