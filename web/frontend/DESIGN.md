# Pack Web 前端设计文档

> 基于 Vue.js 的箱货检测与SKU匹配Web应用前端实现

***

## 1. 前端技术栈

| 技术           | 版本     | 用途           |
| ------------ | ------ | ------------ |
| Vue.js       | 3.x    | 前端框架，响应式数据绑定 |
| Element Plus | Latest | UI组件库        |
| Pinia        | Latest | 状态管理         |
| Axios        | Latest | HTTP客户端      |
| Vite         | 5.x    | 构建工具         |

***

## 2. 目录结构

```
web/frontend/
├── src/
│   ├── App.vue              # 根组件，主页面布局
│   ├── main.js              # 入口文件
│   ├── api/
│   │   └── detector.js      # 检测API调用
│   ├── components/
│   │   ├── StatusBanner.vue    # 状态横幅
│   │   ├── UploadArea.vue      # 上传区域
│   │   ├── ImagePreview.vue    # 图片预览
│   │   ├── DetectionList.vue   # 检测列表
│   │   ├── StatsCard.vue       # 统计卡片
│   │   └── UploadArea.vue      # 上传区域
│   ├── stores/
│   │   └── app.js           # Pinia状态管理
│   └── style.css            # 全局样式
├── package.json
└── vite.config.js
```

***

## 3. 页面结构

```
┌────────────────────────────────────────────────────────────┐
│  Header: Pack Web - 箱货检测与SKU匹配                        │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │              图片上传区域（点击/拖拽）                 │  │
│  │                   📤                                 │  │
│  │              点击或拖拽上传图片                       │  │
│  │              支持 JPG、PNG 格式                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ 图片预览（上传后显示）─────────────────────────────┐   │
│  │ [缩略图]  filename.jpg              [移除]        │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  [🔍 开始检测]  [🔄 重置]                                  │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   检测结果统计                        │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │  │
│  │  │ 检测数量│  │ 已匹配 │  │ 低置信 │  │ 未匹配 │    │  │
│  │  │   12   │  │   6    │  │   2    │  │   4    │    │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    检测结果图片                        │  │
│  │           [带检测框的可视化图像，居中显示]             │  │
│  │       绿框=高置信匹配 黄框=低置信 红框=未匹配         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  📋 检测详情                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ [裁剪图] #1  conf:95.2%  SKU:000001  sim:98.5%  ✅  │  │
│  │ [裁剪图] #2  conf:87.3%  SKU:000003  sim:92.1%  ✅  │  │
│  │ [裁剪图] #3  conf:76.8%  SKU:000007  sim:86.2%  ⚠️  │  │
│  │ [裁剪图] #4  conf:82.1%  Unknown                 ❌  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

***

## 4. 状态机设计

### 4.1 状态定义

| 状态            | 说明        | UI表现           |
| ------------- | --------- | -------------- |
| `IDLE`        | 初始状态      | 上传区域可点击，显示拖拽提示 |
| `UPLOADED`    | 图片已上传，待检测 | 预览图+文件名+检测按钮   |
| `PROCESSING`  | 检测处理中     | Loading动画，按钮禁用 |
| `SUCCESS`     | 检测成功      | 统计卡片+结果图+详情列表  |
| `ERROR`       | 检测失败      | 错误提示+重试按钮      |
| `SYSTEM_INIT` | 系统初始化中    | 红色横幅+禁用所有操作    |

### 4.2 状态流转

```
┌─────────┐    上传图片    ┌─────────┐   点击检测   ┌─────────┐
│  IDLE   │──────────────→│UPLOADED │────────────→│PROCESSING│
└─────────┘               └─────────┘             └────┬────┘
     ↑                       │                          │
     │         点击移除       │                     成功/失败
     └───────────────────────┘                          ↓
                                                 ┌─────────────┐
                                                 │   SUCCESS   │
                                                 └─────────────┘
```

***

## 5. 核心组件

### 5.1 App.vue（根组件）

**职责**：

- 页面整体布局
- 状态管理协调
- API调用触发

**关键逻辑**：

```javascript
const processImage = async () => {
  const result = await detectAndMatch(file)
  store.completeSuccess(result)
}
```

### 5.2 StatusBanner.vue（状态横幅）

**职责**：

- 显示系统状态横幅
- 系统初始化中显示红色警告

### 5.3 stores/app.js（Pinia状态管理）

**状态定义**：

```javascript
{
  systemStatus: 'SYSTEM_INIT' | 'IDLE' | 'UPLOADED' | 'PROCESSING' | 'SUCCESS' | 'ERROR',
  selectedFile: File | null,
  previewUrl: string | null,
  result: DetectResult | null,
  error: string | null
}
```

***

## 6. API接口

### 6.1 检测接口

**端点**：`POST /api/detect-and-match`

**请求**：

```javascript
const formData = new FormData()
formData.append('file', file)
const response = await axios.post('/api/detect-and-match', formData)
```

**响应**：

```json
{
  "success": true,
  "count": 12,
  "matched_count": 6,
  "low_conf_count": 2,
  "unmatched_count": 4,
  "boxes": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.952,
      "class_name": "Carton"
    }
  ],
  "crops": ["base64...", "base64..."],
  "image_with_boxes": "base64...",
  "matches": [
    {
      "sku_id": "000001",
      "similarity": 0.985,
      "status": "matched"
    }
  ]
}
```

### 6.2 健康检查接口

**端点**：`GET /api/health`

**响应**：

```json
{
  "detector_ready": true,
  "matcher_ready": false,
  "sku_count": 0
}
```

***

## 7. 样式设计

### 7.1 主题色彩

| 用途 | 颜色        |
| -- | --------- |
| 主色 | `#667eea` |
| 渐变 | `#764ba2` |
| 成功 | `#67c23a` |
| 警告 | `#e6a23c` |
| 危险 | `#f56c6c` |
| 信息 | `#909399` |

### 7.2 图片展示样式

**结果图片**：

```css
.image-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  background: #f9f9f9;
  border-radius: 8px;
  padding: 10px;
  min-height: 200px;
}

.result-image {
  max-width: 100%;
  max-height: 500px;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 8px;
}
```

**图片裁剪图**：

```css
.thumb {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 6px;
}
```

***

## 8. 启动命令

```bash
# 安装依赖
cd web/frontend
npm install

# 开发模式
npm run dev

# 生产构建
npm run build
```

***

## 9. 已知问题与优化

### 9.1 已解决

- ✅ 图片过大问题：使用 `max-width: 100%; max-height: 500px;` 约束
- ✅ 可视化图片：使用 YOLO 自带的 `plot()` 方法

### 9.2 待优化

- [ ] 支持多图批量上传
- [ ] 移动端响应式适配
- [ ] 结果导出功能（Excel/CSV）

***

*文档更新时间：2026-05-05*
