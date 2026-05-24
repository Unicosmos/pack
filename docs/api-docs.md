# Pack Web API 接口文档

## 1. 概述

本文档描述 Pack Web 项目的 RESTful API 接口规范，用于地堆箱货检测和SKU匹配服务。

**基础路径**: `/api`  
**版本**: v2.0.0  
**协议**: HTTP/HTTPS  
**认证**: 暂未启用（后续计划集成JWT）

---

## 2. 错误码定义

| 错误码 | 含义 | 说明 |
|-------|------|------|
| 200 | 成功 | 请求成功处理 |
| 400 | 请求参数错误 | 参数验证失败或格式错误 |
| 404 | 资源不存在 | 请求的资源未找到 |
| 422 | 验证错误 | FastAPI自动验证失败 |
| 500 | 服务器内部错误 | 服务端处理异常 |
| 503 | 服务不可用 | 模型未加载或服务未就绪 |

**错误响应格式**:
```json
{
  "success": false,
  "detail": "错误描述",
  "status_code": 400
}
```

---

## 3. 接口列表

### 3.1 系统管理

#### 3.1.1 健康检查

**路径**: `GET /health`

**响应**:
```json
{
  "status": "ready",
  "message": "系统正常运行",
  "detector_ready": true,
  "matcher_ready": true,
  "sku_count": 150,
  "model_path": "models/yolov8n.pt",
  "sku_dir": "SKU/sku_library"
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| status | string | 系统状态 (init/ready/error) |
| detector_ready | bool | 检测模型是否就绪 |
| matcher_ready | bool | 匹配器是否就绪 |
| sku_count | int | SKU数量 |

---

### 3.2 检测服务

#### 3.2.1 仅检测

**路径**: `POST /detect`

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|-------|------|
| file | File | ✅ | - | 图片文件 (jpg/png/bmp) |
| conf_threshold | float | ❌ | 0.5 | 置信度阈值 (0-1) |

**响应**:
```json
{
  "success": true,
  "count": 3,
  "boxes": [
    {
      "bbox": [100, 50, 200, 150],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "box"
    }
  ],
  "crops": ["base64..."],
  "image_with_boxes": "base64..."
}
```

#### 3.2.2 检测+匹配（主接口）

**路径**: `POST /detect-and-match`

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|-------|------|
| file | File | ✅ | - | 图片文件 |
| conf_threshold | float | ❌ | 0.5 | 检测置信度阈值 |
| match_threshold | float | ❌ | 0.85 | 匹配相似度阈值 |

**响应**:
```json
{
  "success": true,
  "count": 3,
  "matched_count": 2,
  "low_conf_count": 0,
  "unmatched_count": 1,
  "boxes": [...],
  "matches": [...],
  "image_with_boxes": "base64..."
}
```

---

### 3.3 任务管理

#### 3.3.1 上传图片创建任务

**路径**: `POST /tasks/upload`

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| file | File | ✅ | 图片文件 |

**响应**:
```json
{
  "id": 1,
  "image_name": "test.jpg",
  "status": "pending",
  "detection_status": "pending",
  "review_status": "pending",
  "box_count": 0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### 3.3.2 获取任务列表

**路径**: `GET /tasks`

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|-----|------|------|-------|------|
| page | int | ❌ | 1 | 页码 |
| page_size | int | ❌ | 10 | 每页数量 |
| status_filter | string | ❌ | - | 状态筛选 |

**响应**:
```json
{
  "success": true,
  "tasks": [...],
  "total": 100,
  "page": 1,
  "page_size": 10
}
```

#### 3.3.3 获取任务详情

**路径**: `GET /tasks/{task_id}`

**路径参数**:
| 参数 | 类型 | 说明 |
|-----|------|------|
| task_id | int | 任务ID |

**响应**: TaskResponse

#### 3.3.4 执行检测

**路径**: `POST /tasks/{task_id}/detect`

**路径参数**: task_id (int)

**查询参数**:
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| match_threshold | float | 0.85 | 匹配阈值 |

#### 3.3.5 审核检测结果

**路径**: `PUT /tasks/{task_id}/review`

**请求体**:
```json
{
  "boxes": [
    {
      "box_id": "box_0",
      "status": "approved",
      "custom_sku": null
    }
  ]
}
```

**响应**:
```json
{
  "success": true,
  "task_id": 1,
  "approved_count": 2,
  "rejected_count": 1,
  "message": "审核完成"
}
```

#### 3.3.6 SKU匹配

**路径**: `POST /tasks/{task_id}/match`

**路径参数**: task_id (int)

**查询参数**:
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| match_threshold | float | 0.85 | 匹配阈值 |

#### 3.3.7 删除任务

**路径**: `DELETE /tasks/{task_id}`

#### 3.3.8 批量创建任务

**路径**: `POST /tasks/batch`

**参数**: files (List[UploadFile])

#### 3.3.9 获取任务统计

**路径**: `GET /tasks/stats/summary`

**响应**:
```json
{
  "success": true,
  "total": 100,
  "completed": 80,
  "pending": 20,
  "detected": 15,
  "reviewed": 10,
  "failed": 5,
  "total_detections": 350
}
```

---

### 3.4 SKU管理

#### 3.4.1 获取SKU列表

**路径**: `GET /skus`

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| page | int | 1 | 页码 |
| page_size | int | 20 | 每页数量 |
| search | string | - | 搜索关键词 |
| category | string | - | 分类筛选 |
| status | string | - | 状态筛选 |

**响应**:
```json
{
  "success": true,
  "skus": [...],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

#### 3.4.2 创建SKU

**路径**: `POST /skus`

**请求体**:
```json
{
  "sku_id": "SKU001",
  "sku_name": "商品名称",
  "description": "描述",
  "category": "分类",
  "tags": "标签1,标签2"
}
```

**响应**: SKUResponse

#### 3.4.3 更新SKU

**路径**: `PUT /skus/{sku_id}`

**请求体**:
```json
{
  "sku_name": "新名称",
  "status": "active"
}
```

#### 3.4.4 删除SKU

**路径**: `DELETE /skus/{sku_id}`

**说明**: 软删除，标记is_deleted=true

#### 3.4.5 获取SKU统计

**路径**: `GET /skus/stats`

**响应**:
```json
{
  "success": true,
  "total_skus": 150,
  "active_skus": 140,
  "inactive_skus": 10,
  "total_images": 450
}
```

#### 3.4.6 导入SKU

**路径**: `POST /skus/import`

**参数**: file (CSV文件)

**CSV格式**:
```csv
sku_id,sku_name,description,category,tags
SKU001,商品1,描述1,分类1,标签1
SKU002,商品2,描述2,分类2,标签2
```

#### 3.4.7 上传SKU图片

**路径**: `POST /skus/{sku_id}/images/upload`

**参数**: files (List[UploadFile])

---

### 3.5 SKU审核

#### 3.5.1 获取文件夹列表

**路径**: `GET /sku-review/folders`

#### 3.5.2 获取文件夹图片

**路径**: `GET /sku-review/folder-images/{folder_name}`

#### 3.5.3 获取SKU列表

**路径**: `GET /sku-review/skus?keyword=xxx`

#### 3.5.4 分配图片到SKU

**路径**: `POST /sku-review/assign-images`

**请求体**:
```json
{
  "sku_id": "SKU001",
  "image_paths": ["path1.jpg", "path2.jpg"]
}
```

#### 3.5.5 创建SKU

**路径**: `POST /sku-review/create-sku?name=xxx`

#### 3.5.6 保存数据库

**路径**: `POST /sku-review/save-database`

---

### 3.6 操作日志

#### 3.6.1 获取日志列表

**路径**: `GET /logs`

**参数**:
| 参数 | 类型 | 说明 |
|-----|------|------|
| entity_type | string | 实体类型 |
| entity_id | int | 实体ID |
| action | string | 操作类型 |
| page | int | 页码 |
| page_size | int | 每页数量 |

---

## 4. 数据模型

### 4.1 BoxInfo（检测框信息）

| 字段 | 类型 | 说明 |
|-----|------|------|
| bbox | list[float] | 检测框坐标 [x1, y1, x2, y2] |
| confidence | float | 置信度 (0-1) |
| class_id | int | 类别ID |
| class_name | string | 类别名称 |

### 4.2 MatchInfo（匹配信息）

| 字段 | 类型 | 说明 |
|-----|------|------|
| sku_id | string | 匹配的SKU编号 |
| similarity | float | 相似度 (0-1) |
| ratio | float | 比例 |
| status | string | 匹配状态 (matched/low_conf/unmatched) |
| top5_labels | list | Top5候选 |

### 4.3 TaskResponse（任务响应）

| 字段 | 类型 | 说明 |
|-----|------|------|
| id | int | 任务ID |
| image_name | string | 图片名称 |
| status | string | 主状态 |
| detection_status | string | 检测状态 |
| review_status | string | 审核状态 |
| box_count | int | 检测框数量 |
| matched_count | int | 已匹配数量 |
| unmatched_count | int | 未匹配数量 |
| result | object | 检测结果 |
| created_at | string | 创建时间 |
| completed_at | string | 完成时间 |

### 4.4 SKUResponse（SKU响应）

| 字段 | 类型 | 说明 |
|-----|------|------|
| id | int | 数据库ID |
| sku_id | string | SKU编号 |
| sku_name | string | SKU名称 |
| description | string | 描述 |
| category | string | 分类 |
| status | string | 状态 (active/inactive) |
| image_count | int | 图片数量 |
| tags | string | 标签 |
| created_at | string | 创建时间 |
| updated_at | string | 更新时间 |

---

## 5. 使用示例

### 5.1 JavaScript客户端示例

```javascript
// 检测并匹配图片
const formData = new FormData()
formData.append('file', file)
formData.append('conf_threshold', 0.5)
formData.append('match_threshold', 0.85)

const response = await fetch('/api/detect-and-match', {
  method: 'POST',
  body: formData
})
const result = await response.json()

// 获取任务列表
const response = await fetch('/api/tasks?page=1&page_size=10')
const data = await response.json()
```

### 5.2 Python客户端示例

```python
import requests

# 检测图片
with open('test.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/detect-and-match',
        files={'file': f},
        data={'conf_threshold': 0.5, 'match_threshold': 0.85}
    )
result = response.json()

# 获取任务列表
response = requests.get('http://localhost:8000/api/tasks', params={'page': 1})
data = response.json()
```

---

## 6. 版本历史

| 版本 | 日期 | 变更说明 |
|-----|------|---------|
| v2.0.0 | 2024-01-15 | 重构版，新增任务管理、SKU审核流程 |
| v1.0.0 | 2023-12-01 | 初始版本，基础检测功能 |
