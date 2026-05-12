# API 接口文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **内容类型**: `multipart/form-data`（上传文件）或 `application/json`

---

## 1. 健康检查

### GET /api/health

检查系统运行状态。

**响应示例**：
```json
{
  "status": "ok",
  "message": "系统正常运行",
  "detector_ready": true,
  "matcher_ready": true,
  "sku_count": 197,
  "model_path": "D:/A_pack/pack/data/models/yolo/best.pt",
  "sku_dir": "D:/A_pack/pack/data/sku_library"
}
```

**状态说明**：
| status | 说明 |
|--------|------|
| `ok` | 完全正常运行 |
| `partial` | 检测就绪，匹配不可用 |
| `error` | 检测模型加载失败 |
| `init` | 系统初始化中 |

---

## 2. 图片检测

### POST /api/detect

仅进行目标检测，不进行SKU匹配。

**请求参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| conf_threshold | float | 否 | 置信度阈值（默认0.5） |

**响应示例**：
```json
{
  "success": true,
  "count": 3,
  "boxes": [
    {
      "bbox": [120, 80, 340, 280],
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "box"
    }
  ],
  "crops": ["base64编码的裁剪图..."],
  "image_with_boxes": "base64编码的带框图..."
}
```

---

## 3. SKU匹配

### POST /api/match

仅进行SKU匹配（需要已检测的裁剪图）。

**请求参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| match_threshold | float | 否 | 匹配阈值（默认0.85） |
| ratio_threshold | float | 否 | Ratio Test阈值（默认1.2） |

**响应示例**：
```json
{
  "success": true,
  "sku_id": "000005",
  "similarity": 0.94,
  "ratio": 1.45,
  "status": "matched",
  "top5_labels": [
    {
      "label": "1 (143)_001",
      "similarity": 0.94,
      "image_name": "1 (143)_001.jpg",
      "sku_id": "000005",
      "sku_name": "商品名称"
    }
  ]
}
```

---

## 4. 检测+匹配（主接口）

### POST /api/detect-and-match

同时进行目标检测和SKU匹配。

**请求参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| conf_threshold | float | 否 | 置信度阈值（默认0.35） |
| match_threshold | float | 否 | 匹配阈值（默认0.85） |

**响应示例**：
```json
{
  "success": true,
  "count": 2,
  "matched_count": 1,
  "low_conf_count": 0,
  "unmatched_count": 1,
  "boxes": [...],
  "crops": [...],
  "image_with_boxes": "base64...",
  "matches": [
    {
      "sku_id": "000005",
      "similarity": 0.94,
      "ratio": 1.45,
      "status": "matched",
      "top5_labels": [...]
    },
    null
  ],
  "sku_matcher_enabled": true
}
```

**匹配状态说明**：
| status | 说明 |
|--------|------|
| `matched` | 匹配成功，相似度高于阈值 |
| `low_conf` | 相似度较低，可能需要人工确认 |
| `unmatched` | 未匹配到合适的SKU |

---

## 5. SKU列表

### GET /api/skus

获取SKU库中的所有SKU列表。

**响应示例**：
```json
{
  "success": true,
  "count": 65,
  "skus": [
    {
      "sku_id": "000002",
      "sku_name": "商品名称",
      "label_count": 3,
      "image_count": 3
    }
  ]
}
```

---

## 6. SKU图片

### GET /api/sku-image/{sku_id}/{image_name}

获取SKU库中的指定图片。

**路径参数**：
| 参数 | 说明 |
|------|------|
| sku_id | SKU编号（如000005） |
| image_name | 图片名称（需URL编码） |

**示例**：
```
GET /api/sku-image/000005/1%20%28143%29_001.jpg
```

**响应**：图片二进制数据

---

## 错误响应

所有接口的错误响应格式：

```json
{
  "success": false,
  "detail": "错误描述信息",
  "status_code": 404
}
```

**常见错误码**：
| 状态码 | 说明 |
|--------|------|
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用（如模型未加载） |

---

## 前端调用示例

### 使用fetch API

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/api/detect-and-match', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

### 使用Vue.js + Axios

```javascript
import axios from 'axios';

const formData = new FormData();
formData.append('file', file);

const response = await axios.post('/api/detect-and-match', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});
```
