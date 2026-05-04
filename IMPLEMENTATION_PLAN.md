# Pack Web 项目完善规划

> 基于 `DESIGN.md` 的详细实施计划
>
> 更新时间：2026-05-04

---

## 一、项目现状分析

### 1.1 已有文件

```
pack/
├── web/
│   ├── backend/
│   │   ├── main.py              # FastAPI入口（基础）
│   │   ├── create_test_image.py # 测试图片生成
│   │   └── static/
│   │       └── index.html       # 纯HTML+JS前端（极简版）
│   └── frontend/               # Vue3前端骨架（未完成）
│
├── YOLO/
│   ├── pack_train.py           # ✅ 已有
│   ├── pack_train_occlusion_aug.py  # ✅ 已有
│   ├── pack_predict.py         # ✅ 已有
│   └── pack_val.py            # ✅ 已有
│
├── SKU/
│   ├── sku_matcher.py         # ⚠️ 需要修改（实现Ratio Test两步验证）
│   ├── sku_augmentation.py    # ✅ 已有
│   ├── sku_review.py          # ✅ 已有（Gradio）
│   └── smart_sku_manager.py   # ✅ 已有
│
├── utils/
│   ├── box_detector.py        # ⚠️ 需要修改（增加小框过滤）
│   └── ...                    # 其他工具已有
│
├── configs/                    # ❌ 缺失（需创建）
├── models/                    # ❌ 待放置模型
└── sku_library/               # ❌ 待创建（CSV+npy格式）
```

### 1.2 当前完成度评估

| 模块 | 完成度 | 说明 |
|------|--------|------|
| 后端API框架 | 60% | 基础框架完成，缺少错误处理和状态管理 |
| 前端 | 30% | 纯HTML版可用，Vue版需重建 |
| SKU匹配 | 40% | 需要实现Ratio Test两步验证 |
| SKU库 | 0% | 格式不符，需按设计重构 |
| 配置管理 | 0% | 缺失 |

---

## 二、需要添加的文件

### 2.1 Web后端新增

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `web/backend/config.py` | 后端配置管理（检测/匹配阈值、路径配置） | P0 |
| `web/backend/models/schemas.py` | Pydantic数据模型（请求/响应格式） | P0 |
| `web/backend/api/v1/detector.py` | 检测API路由 | P1 |
| `web/backend/api/v1/matcher.py` | 匹配API路由 | P1 |
| `web/backend/api/v1/skus.py` | SKU库API路由 | P1 |
| `web/backend/api/v1/__init__.py` | API v1初始化 | P1 |
| `web/backend/core/detector.py` | BoxDetector封装 | P0 |
| `web/backend/core/matcher.py` | SKUMatcher封装（含Ratio Test） | P0 |
| `web/backend/core/visualizer.py` | 结果可视化模块 | P1 |
| `web/backend/core/__init__.py` | Core模块初始化 | P1 |
| `web/backend/utils/image_utils.py` | 图像处理工具（小框过滤、resize） | P0 |
| `web/backend/utils/__init__.py` | Utils初始化 | P1 |
| `web/backend/dependencies.py` | FastAPI依赖注入 | P2 |
| `web/backend/constants.py` | 常量定义 | P2 |

### 2.2 Web前端新增（Vue.js SPA）

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `web/frontend/package.json` | Node依赖配置 | P0 |
| `web/frontend/vite.config.js` | Vite构建配置 | P0 |
| `web/frontend/index.html` | 入口HTML | P0 |
| `web/frontend/src/main.js` | Vue应用入口 | P0 |
| `web/frontend/src/App.vue` | 根组件 | P0 |
| `web/frontend/src/router/index.js` | Vue Router配置 | P1 |
| `web/frontend/src/stores/app.js` | Pinia状态管理 | P1 |
| `web/frontend/src/api/detector.js` | 检测API调用 | P0 |
| `web/frontend/src/api/skus.js` | SKU API调用 | P1 |
| `web/frontend/src/components/UploadArea.vue` | 图片上传组件 | P0 |
| `web/frontend/src/components/ImagePreview.vue` | 图片预览组件 | P0 |
| `web/frontend/src/components/StatsCard.vue` | 统计卡片组件 | P0 |
| `web/frontend/src/components/DetectionList.vue` | 检测列表组件 | P0 |
| `web/frontend/src/components/StatusBanner.vue` | 状态横幅组件 | P1 |
| `web/frontend/src/components/LoadingSpinner.vue` | 加载动画组件 | P1 |
| `web/frontend/src/views/HomeView.vue` | 首页 | P0 |
| `web/frontend/src/views/ResultView.vue` | 结果页 | P1 |
| `web/frontend/src/styles/common.css` | 公共样式 | P1 |
| `web/frontend/.env` | 环境变量 | P2 |
| `web/frontend/.gitignore` | Git忽略配置 | P1 |

### 2.3 配置文件新增

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `configs/app_config.yaml` | 应用配置（检测阈值、匹配阈值、路径） | P0 |
| `configs/model_config.yaml` | 模型配置 | P1 |
| `configs/sku_config.yaml` | SKU库配置 | P1 |

### 2.4 SKU库结构新增

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `sku_library/sku_library.csv` | 图片-标签映射（核心索引） | P0 |
| `sku_library/sku_features.npy` | 特征矩阵 [N, 384] | P0 |
| `sku_library/images/000001/` | SKU图片存储目录 | P0 |
| `sku_library/labels/` | OML标签文件 | P2 |

### 2.5 文档新增

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `web/IMPLEMENTATION_PLAN.md` | 本文档 | - |

---

## 三、需要修改的文件

### 3.1 后端修改

| 文件路径 | 修改内容 | 优先级 |
|----------|----------|--------|
| `web/backend/main.py` | 1. 添加 `lifespan` 生命周期管理<br>2. 添加 SYSTEM_INIT 状态处理<br>3. 完善错误处理（503等）<br>4. 添加 `/api/match` 端点<br>5. 集成新的 `core/` 模块 | P0 |
| `web/backend/requirements.txt` | 添加 `open-metric-learning` 依赖 | P0 |

### 3.2 SKU模块修改

| 文件路径 | 修改内容 | 优先级 |
|----------|----------|--------|
| `SKU/sku_matcher.py` | 1. 实现 Ratio Test 两步验证逻辑<br>2. 添加 `is_ready()` 方法<br>3. 返回 `top5_labels` 信息<br>4. 支持 CSV 格式索引 | P0 |
| `SKU/box_detector.py` (在utils中) | 1. 添加小框过滤逻辑<br>2. 添加 `resize_with_padding` 函数<br>3. 输出分割掩码 | P0 |
| `SKU/feature_extractor.py` | 1. 实现 ViT-S16 DINO 特征提取<br>2. 添加 L2 归一化<br>3. 支持微调模型回退预训练模型 | P1 |

### 3.3 前端修改

| 文件路径 | 修改内容 | 优先级 |
|----------|----------|--------|
| `web/frontend/src/App.vue` | 完全重写为 Vue3 Composition API | P0 |
| `web/frontend/src/main.js` | 适配新的项目结构 | P0 |
| `web/frontend/index.html` | 适配Vue Router | P1 |

---

## 四、新增模块详细说明

### 4.1 `web/backend/core/detector.py` - BoxDetector封装

```python
# 功能：
# 1. 封装YOLO检测逻辑
# 2. 支持自定义置信度阈值
# 3. 返回标准化的检测结果格式

class BoxDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5)
    def detect_single_image(self, image: PIL.Image.Image) -> List[Dict]
    def is_ready(self) -> bool
```

### 4.2 `web/backend/core/matcher.py` - SKUMatcher封装

```python
# 功能：
# 1. 加载sku_features.npy和sku_library.csv
# 2. 实现Ratio Test两步验证
# 3. 返回MatchResult包含：status, sku_id, similarity, ratio, top5_labels

class SKUMatcher:
    def __init__(self, model_path: str, sku_dir: str)
    def match_sku(self, query: np.ndarray, threshold: float = 0.85, ratio_threshold: float = 1.2) -> MatchResult
    def is_ready(self) -> bool
    def get_sku_info(self, sku_id: str) -> Dict
```

### 4.3 `web/backend/core/visualizer.py` - 结果可视化

```python
# 功能：
# 1. 在原图上绘制检测框
# 2. 颜色编码：绿=matched, 黄=low_conf, 红=unmatched, 灰=无匹配
# 3. 生成Base64编码

def draw_detection_result(
    image: PIL.Image.Image,
    boxes: List[Dict],
    match_results: List[MatchResult] | None = None
) -> Tuple[PIL.Image.Image, List[str]]
```

### 4.4 `web/backend/utils/image_utils.py` - 图像处理工具

```python
# 功能：
# 1. 小框过滤：面积占比<1%、像素<2500、宽高比异常
# 2. resize_with_padding：224×224，保持比例，灰色填充
# 3. 图像格式转换工具

def filter_small_boxes(boxes: List[Dict], image_size: Tuple) -> List[Dict]
def resize_with_padding(image: PIL.Image.Image, target_size: int = 224) -> PIL.Image.Image
def crop_box(image: PIL.Image.Image, bbox: List[int]) -> PIL.Image.Image
```

### 4.5 `web/backend/config.py` - 配置管理

```python
# 功能：
# 1. 加载configs/app_config.yaml
# 2. 提供检测阈值、匹配阈值、路径配置
# 3. 单例模式

class Config:
    CONF_THRESHOLD: float = 0.5
    MATCH_THRESHOLD: float = 0.85
    RATIO_THRESHOLD: float = 1.2
    MODEL_PATH: Path
    SKU_DIR: Path
```

### 4.6 `web/backend/models/schemas.py` - Pydantic模型

```python
# 请求模型
class DetectRequest(BaseModel):
    pass  # FastAPI自动处理multipart

# 响应模型
class BoxInfo(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str

class MatchInfo(BaseModel):
    sku_id: Optional[str]
    similarity: Optional[float]
    ratio: Optional[float]
    status: str  # matched/low_conf/unmatched
    top5_labels: Optional[List[Dict]]

class DetectAndMatchResponse(BaseModel):
    success: bool
    count: int
    matched_count: int
    low_conf_count: int
    unmatched_count: int
    boxes: List[BoxInfo]
    crops: List[str]
    image_with_boxes: str
    matches: List[MatchInfo]
```

---

## 五、API端点实现规划

### 5.1 需要实现的端点

| 端点 | 方法 | 文件位置 | 说明 |
|------|------|----------|------|
| `/api/health` | GET | main.py | 健康检查，包含系统就绪状态 |
| `/api/detect` | POST | api/v1/detector.py | 仅检测 |
| `/api/detect-and-match` | POST | api/v1/detector.py | 检测+匹配 |
| `/api/match` | POST | api/v1/matcher.py | 仅SKU匹配 |
| `/api/skus` | GET | api/v1/skus.py | SKU列表 |
| `/api/skus/{id}` | GET | api/v1/skus.py | SKU详情 |

### 5.2 API路由结构

```
web/backend/
├── main.py                    # FastAPI入口，注册路由
├── api/
│   └── v1/
│       ├── __init__.py
│       ├── detector.py        # /api/detect, /api/detect-and-match
│       ├── matcher.py         # /api/match
│       └── skus.py           # /api/skus, /api/skus/{id}
```

---

## 六、前端Vue组件规划

### 6.1 组件结构

```
web/frontend/src/
├── components/
│   ├── UploadArea.vue       # 上传区域（拖拽+点击）
│   ├── ImagePreview.vue      # 图片预览（缩略图+文件名）
│   ├── StatsCard.vue        # 统计卡片（4个数字）
│   ├── DetectionList.vue     # 检测列表（裁剪图+详情）
│   ├── StatusBanner.vue      # 状态横幅（SYSTEM_INIT红色等）
│   └── LoadingSpinner.vue    # 加载动画
├── views/
│   ├── HomeView.vue          # 首页（上传+结果）
│   └── ResultView.vue        # 结果页
├── stores/
│   └── app.js               # Pinia状态管理
└── router/
    └── index.js             # 路由配置
```

### 6.2 状态机实现

```javascript
// stores/app.js
const states = ['IDLE', 'UPLOADED', 'PROCESSING', 'SUCCESS', 'ERROR', 'SYSTEM_INIT']

const state = reactive({
    currentState: 'IDLE',
    selectedFile: null,
    result: null,
    error: null,
    skuCount: 0
})

// 状态流转
function uploadImage(file) { ... }
function startDetection() { ... }
function reset() { ... }
```

---

## 七、实施优先级与时间估算

### 7.1 Phase 1：后端核心（P0）

| 任务 | 文件 | 优先级 | 预估 |
|------|------|--------|------|
| 配置管理模块 | `config.py` | P0 | 0.5h |
| Pydantic模型 | `models/schemas.py` | P0 | 0.5h |
| BoxDetector封装 | `core/detector.py` | P0 | 1h |
| SKUMatcher封装（含Ratio Test） | `core/matcher.py` | P0 | 2h |
| 图像处理工具 | `utils/image_utils.py` | P0 | 1h |
| 可视化模块 | `core/visualizer.py` | P1 | 1h |
| 重构main.py | `main.py` | P0 | 1h |

**Phase 1 完成后**：后端API可独立运行，匹配逻辑正确

### 7.2 Phase 2：前端核心（P0）

| 任务 | 文件 | 优先级 | 预估 |
|------|------|--------|------|
| 项目脚手架 | `package.json`, `vite.config.js`, `main.js` | P0 | 0.5h |
| 根组件 | `App.vue` | P0 | 1h |
| 上传组件 | `UploadArea.vue` | P0 | 1h |
| 预览组件 | `ImagePreview.vue` | P0 | 0.5h |
| 统计卡片 | `StatsCard.vue` | P0 | 0.5h |
| 检测列表 | `DetectionList.vue` | P0 | 1h |
| API调用层 | `api/detector.js` | P0 | 0.5h |

**Phase 2 完成后**：完整的前端页面可运行

### 7.2.1 Phase 2 完成记录 (2026-05-04)

| 任务 | 状态 | 说明 |
|------|------|------|
| 项目脚手架 | ✅ 完成 | package.json, vite.config.js, main.js, index.html |
| 根组件 | ✅ 完成 | App.vue 重构，使用组件化结构 |
| 上传组件 | ✅ 完成 | UploadArea.vue 支持拖拽/点击上传 |
| 预览组件 | ✅ 完成 | ImagePreview.vue 图片预览显示 |
| 统计卡片 | ✅ 完成 | StatsCard.vue 支持颜色和数值显示 |
| 检测列表 | ✅ 完成 | DetectionList.vue 完整检测详情 |
| API调用层 | ✅ 完成 | api/detector.js 封装API调用 |
| npm依赖安装 | ✅ 完成 | 73个包安装成功 |
| 前端运行 | ✅ 完成 | http://localhost:5173 |
| 后端运行 | ✅ 完成 | http://localhost:8000 |

### 7.3 Phase 3：状态与错误处理（P1）✅

| 任务 | 文件 | 优先级 | 预估 | 状态 |
|------|------|--------|------|------|
| 状态横幅组件 | `StatusBanner.vue` | P1 | 0.5h | ✅ |
| Pinia状态管理 | `stores/app.js` | P1 | 1h | ✅ |
| SYSTEM_INIT处理 | main.py + 前端 | P1 | 1h | ✅ |
| 错误处理完善 | main.py | P1 | 1h | ✅ |
| 空状态UI | App.vue | P1 | 0.5h | ✅ |

**Phase 3 完成后**：完整的状态机和错误处理

#### 7.3.1 Phase 3 完成记录 (2026-05-04)

| 任务 | 说明 |
|------|------|
| 状态横幅组件 | 支持init/error/no-sku三种状态显示不同颜色横幅 |
| Pinia状态管理 | 实现完整状态机：IDLE → UPLOADED → PROCESSING → SUCCESS/ERROR |
| SYSTEM_INIT处理 | health接口返回status: init/partial/ok/error四种状态 |
| 全局异常处理 | 统一处理HTTPException、ValidationError、通用异常 |
| 空状态UI | 无结果时显示友好的空状态提示 |
| 新增API端点 | `/api/match` 仅SKU匹配接口 |

### 7.4 Phase 4：SKU库与工具（P0）

| 任务 | 文件 | 优先级 | 预估 |
|------|------|--------|------|
| 创建sku_library目录结构 | - | P0 | 0.5h |
| sku_library.csv格式 | - | P0 | 0.5h |
| 特征提取脚本 | SKU/feature_extractor.py | P1 | 2h |
| 建库工具 | SKU/build_library.py (新) | P1 | 2h |

**Phase 4 完成后**：SKU库可正确加载

### 7.5 Phase 5：API路由拆分（P1）

| 任务 | 文件 | 优先级 | 预估 |
|------|------|--------|------|
| API v1目录结构 | `api/v1/` | P1 | 0.5h |
| detector路由 | `api/v1/detector.py` | P1 | 1h |
| matcher路由 | `api/v1/matcher.py` | P1 | 0.5h |
| skus路由 | `api/v1/skus.py` | P1 | 1h |

**Phase 5 完成后**：API结构更清晰

---

## 八、文件清单汇总

### 8.1 新增文件（按优先级）

```
P0 (必须实现):
├── web/backend/
│   ├── config.py
│   ├── models/schemas.py
│   ├── core/detector.py
│   ├── core/matcher.py
│   └── utils/image_utils.py
├── web/frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── src/main.js
│   ├── src/App.vue
│   ├── src/components/UploadArea.vue
│   ├── src/components/ImagePreview.vue
│   ├── src/components/StatsCard.vue
│   ├── src/components/DetectionList.vue
│   └── src/api/detector.js
├── configs/
│   └── app_config.yaml
└── sku_library/
    └── (目录结构)

P1 (重要功能):
├── web/backend/
│   ├── api/v1/__init__.py
│   ├── api/v1/detector.py
│   ├── api/v1/matcher.py
│   ├── api/v1/skus.py
│   └── core/visualizer.py
├── web/frontend/
│   ├── src/router/index.js
│   ├── src/stores/app.js
│   ├── src/components/StatusBanner.vue
│   ├── src/components/LoadingSpinner.vue
│   └── src/styles/common.css
├── SKU/
│   └── feature_extractor.py (修改)
└── configs/
    ├── model_config.yaml
    └── sku_config.yaml

P2 (优化完善):
├── web/backend/
│   ├── dependencies.py
│   └── constants.py
└── web/frontend/
    ├── .env
    └── .gitignore
```

### 8.2 修改文件（按优先级）

```
P0 (必须修改):
├── web/backend/main.py         # 完整重构
├── web/backend/requirements.txt # 添加OML依赖
├── SKU/sku_matcher.py          # 实现Ratio Test两步验证
└── utils/box_detector.py      # 添加小框过滤

P1 (重要修改):
├── web/frontend/src/App.vue    # 重写为Vue3
└── web/frontend/src/main.js   # 适配新结构

P2 (可选修改):
└── SKU/feature_extractor.py    # 实现ViT特征提取
```

---

## 九、依赖清单

### 9.1 Python依赖

```
# requirements.txt (web/backend/)
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
ultralytics>=8.1.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
open-metric-learning>=0.3.0
pyyaml>=6.0.0
```

### 9.2 Node依赖

```json
// package.json (web/frontend/)
{
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.2.0",
    "pinia": "^2.1.0",
    "axios": "^1.6.0",
    "element-plus": "^2.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

---

## 十、验收标准

### 10.1 Phase 1 验收 ✅

- [x] `curl http://localhost:8000/api/health` 返回系统状态
- [x] 后端模块结构创建完成：config.py, models/schemas.py, utils/image_utils.py, core/visualizer.py, core/matcher.py
- [x] main.py 重构完成，集成所有新模块
- [x] Ratio Test两步验证逻辑实现于 core/matcher.py
- [ ] POST `/api/detect` 返回检测结果（需YOLO模型）
- [ ] POST `/api/detect-and-match` 返回完整结果（含matched/low_conf/unmatched）（需模型和SKU库）
- [ ] Ratio Test逻辑正确验证（需SKU库）

### 10.2 Phase 2 验收 ✅

- [x] 前端可正常启动 `npm run dev`
- [x] 创建Vue组件：UploadArea, ImagePreview, StatsCard, DetectionList
- [x] 创建API调用层 api/detector.js
- [x] App.vue使用组件重构完成
- [x] 前端运行在 http://localhost:5173
- [x] 后端运行在 http://localhost:8000
- [x] `/api/health` 返回系统状态
- [ ] 图片上传显示预览（需前端测试）
- [ ] 点击检测后显示结果（需模型文件）
- [ ] 统计卡片数字正确（需检测结果）

### 10.3 Phase 3 验收 ✅

- [x] SYSTEM_INIT状态显示黄色横幅（warning状态）
- [x] 错误状态正确显示错误信息（error状态）
- [x] count=0时显示空状态提示
- [x] SKU库不存在时显示蓝色提示条（no-sku状态）
- [x] Pinia状态管理集成完成
- [x] 全局异常处理完善

### 10.4 Phase 4 验收（SKU库建设）✅

- [x] 修改 box_detector.py 输出结构 → crops/{原图名称}/
- [x] 修改 sku_augmentation.py 生成 sku_library.csv 索引
- [x] 创建 feature_extractor.py（ViT-S16 DINO，384维）
- [x] 特征维度统一为384维（适配ViT-S16 DINO）
- [ ] sku_library.csv格式验证（需实际运行）
- [ ] sku_features.npy形状为[N, 384]（需实际运行）
- [ ] 匹配结果正确返回sku_id和similarity（需SKU库）

---

## 十一、后续迭代（可选）

| 方向 | 内容 | 优先级 |
|------|------|--------|
| 批量处理 | 多图上传、队列处理 | P2 |
| SKU管理界面 | 在线增删SKU | P2 |
| 结果导出 | Excel/CSV导出 | P2 |
| WebSocket | 实时进度推送 | P3 |
