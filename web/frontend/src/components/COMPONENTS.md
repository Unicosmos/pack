# 前端组件库文档

## 目录结构

```
components/
├── pages/                    # 页面级组件（路由直接引用）
│   ├── HomePage.vue          # 首页
│   ├── TaskListPage.vue      # 任务列表页
│   ├── SkuListPage.vue       # SKU列表页
│   └── SkuReviewPage.vue     # SKU审核页
├── layout/                   # 页面布局组件（统一页面结构）
│   ├── PageHeader.vue        # 顶部横幅组件
│   └── PageContainer.vue     # 内容容器组件
├── ui/                       # 通用UI组件（可复用）
│   ├── StatusBanner.vue      # 状态横幅
│   ├── ImageViewer.vue       # 图片查看器
│   ├── ActionMenu.vue        # 操作菜单
│   ├── FilterBar.vue         # 筛选工具栏
│   ├── FilterDropdown.vue    # 筛选下拉框
│   └── TimeFilterDropdown.vue # 时间筛选下拉框
├── task/                     # 任务相关组件
│   ├── BatchResultCard.vue   # 批量结果卡片
│   ├── DetectionList.vue     # 检测结果列表
│   ├── TaskStats.vue         # 任务统计图表
│   ├── TaskNav.vue           # 任务导航侧边栏
│   ├── MatchResult.vue       # 匹配结果卡片（含Top5候选）
│   └── BoxMatchDialog.vue    # 箱体匹配对话框
├── sku/                      # SKU相关组件
│   └── SkuImage.vue          # SKU图片组件
├── upload/                   # 上传相关组件
│   ├── UploadArea.vue        # 上传区域
│   └── FileList.vue          # 文件列表
├── home/                     # 首页专用组件
│   ├── RecentTasks.vue       # 最近任务列表
│   └── SysStatusBar.vue      # 系统状态栏
└── COMPONENTS.md             # 组件文档
```

## 组件分类说明

### 1. 页面组件 (pages/)
页面级组件，作为路由的直接入口，包含完整的页面布局和业务逻辑。

| 组件名        | 说明                     | 依赖                      |
| ------------- | ------------------------ | ------------------------- |
| LoginPage     | 登录页面                 | 无                        |
| HomePage      | 首页，包含上传和检测功能 | UploadArea, DetectionList |
| TaskListPage  | 任务列表管理页面         | ImageViewer, TaskStats    |
| SkuListPage   | SKU库管理页面            | SkuImage, ImageViewer     |
| SkuReviewPage | SKU审核页面              | ImageViewer               |

### 2. 布局组件 (layout/)
页面布局公共组件，用于统一页面结构，解决页面切换时的视觉跳变问题。

| 组件名        | 说明         | 功能特性                         |
| ------------- | ------------ | -------------------------------- |
| PageHeader    | 顶部横幅组件 | 统一导航、响应式设计、状态保持   |
| PageContainer | 内容容器组件 | 标准化布局、侧边栏支持、无缝对接 |

#### 2.1 PageHeader（顶部横幅组件）

**功能说明**：提供统一的页面顶部横幅，包含标题、导航菜单和操作按钮区域。

**Props配置**：

| 属性名          | 类型   | 默认值 | 说明                        |
| --------------- | ------ | ------ | --------------------------- |
| title           | String | 必选   | 页面标题                    |
| logo            | String | ''     | Logo图标（emoji或图标字符） |
| navigationItems | Array  | []     | 导航菜单项数组              |

**导航菜单项结构**：
```javascript
{
  id: 'home',           // 菜单项ID
  label: '首页',        // 显示文本
  icon: '🏠',          // 图标
  href: '/',           // 链接地址
  active: true         // 是否激活状态
}
```

**Slots**：
| 插槽名          | 说明             |
| --------------- | ---------------- |
| navigation      | 自定义导航内容   |
| actions / right | 右侧操作按钮区域 |

**Events**：
| 事件名   | 参数 | 说明             |
| -------- | ---- | ---------------- |
| navigate | item | 导航项点击时触发 |

**使用示例**：
```vue
<PageHeader 
  title="页面标题"
  :navigation-items="navItems"
  @navigate="handleNavigate"
>
  <template #actions>
    <button class="btn btn-primary">操作按钮</button>
  </template>
</PageHeader>
```

#### 2.2 PageContainer（内容容器组件）

**功能说明**：提供标准化的页面内容容器，统一管理边距、内边距和最大宽度。

**Props配置**：

| 属性名          | 类型    | 默认值   | 说明                     |
| --------------- | ------- | -------- | ------------------------ |
| title           | String  | ''       | 内容区域标题             |
| subtitle        | String  | ''       | 内容区域副标题           |
| showHeader      | Boolean | true     | 是否显示头部区域         |
| hasSidebar      | Boolean | false    | 是否包含侧边栏           |
| sidebarPosition | String  | 'left'   | 侧边栏位置（left/right） |
| maxWidth        | String  | '1800px' | 最大宽度                 |
| padding         | String  | '20px'   | 内边距                   |

**Slots**：
| 插槽名         | 说明           |
| -------------- | -------------- |
| default        | 主内容区域     |
| header         | 自定义头部内容 |
| sidebar        | 左侧侧边栏     |
| sidebar-header | 侧边栏头部     |
| sidebar-body   | 侧边栏主体     |
| sidebar-footer | 侧边栏底部     |
| sidebar-right  | 右侧侧边栏     |
| footer         | 底部区域       |

**使用示例**：
```vue
<PageContainer 
  title="内容标题" 
  subtitle="内容描述"
  :has-sidebar="true"
>
  <!-- 主内容 -->
  <div class="content">
    页面内容...
  </div>
  
  <!-- 侧边栏 -->
  <template #sidebar>
    <div class="sidebar-content">
      侧边栏内容...
    </div>
  </template>
</PageContainer>
```

### 3. UI组件 (ui/)
通用UI组件，不包含业务逻辑，可在多个页面复用。

| 组件名       | 说明         | 适用场景     |
| ------------ | ------------ | ------------ |
| StatusBanner | 状态提示横幅 | 全局状态展示 |
| ImageViewer  | 大图查看器   | 图片预览弹窗 |

### 4. 任务组件 (task/)
与任务检测相关的业务组件。

| 组件名          | 说明             | 关联模块                     |
| --------------- | ---------------- | ---------------------------- |
| BatchResultCard | 批量检测结果卡片 | DetectionList                |
| DetectionList   | 检测结果列表     | BatchResultCard, MatchResult |
| TaskStats       | 任务统计图表     | 无                           |
| TaskNav         | 任务导航侧边栏   | 无                           |
| MatchResult     | 匹配结果卡片     | SkuImage                     |

### 5. SKU组件 (sku/)
SKU相关的展示组件。

| 组件名   | 说明        | 特性               |
| -------- | ----------- | ------------------ |
| SkuImage | SKU图片展示 | 支持懒加载、占位符 |

### 6. 上传组件 (upload/)
文件上传相关组件。

| 组件名     | 说明     | 功能               |
| ---------- | -------- | ------------------ |
| UploadArea | 上传区域 | 拖拽上传、点击上传 |
| FileList   | 文件列表 | 已选择文件展示     |

## 命名规范

### 文件命名
- 使用 PascalCase（大驼峰）命名组件文件
- 页面组件：`Page` 后缀（如 `HomePage.vue`）
- 卡片组件：`Card` 后缀（如 `StatsCard.vue`）
- 对话框组件：`Dialog` 后缀

### 组件命名
- 组件名与文件名一致
- 使用 PascalCase 命名
- 避免使用简写或缩写

### 变量命名
- 使用 camelCase（小驼峰）
- 组件属性：以 `props` 对象定义
- 事件：以 `on` 开头（如 `onUpdate`）

## 代码风格规范

### 1. 模板结构
```vue
<template>
  <div class="component-name">
    <!-- 组件内容 -->
  </div>
</template>

<script setup>
// 导入依赖
import { ref, computed } from 'vue'

// Props定义
defineProps({
  propName: {
    type: String,
    required: true,
    default: ''
  }
})

// Emits定义
const emit = defineEmits(['update'])

// 组件逻辑
const state = ref({})
</script>

<style scoped>
.component-name {
  /* 样式 */
}
</style>
```

### 2. 注释规范
- 使用 JSDoc 格式注释组件和方法
- 关键逻辑添加注释说明
- Props 和 Emits 必须有文档注释

### 3. 样式规范
- 使用 scoped 样式
- 使用 CSS 变量（如 `var(--color-primary)`）
- 类名使用 kebab-case

## 公共组件封装建议

### 待封装组件

| 组件名     | 功能         | 优先级 |
| ---------- | ------------ | ------ |
| Button     | 统一按钮样式 | 高     |
| Dropdown   | 统一下拉菜单 | 高     |
| Loader     | 统一加载动画 | 中     |
| EmptyState | 空状态提示   | 中     |
| Pagination | 分页组件     | 中     |

### 待提取 Hooks

| Hook名 | 功能 | 适用组件 |
| ------ | ---- | -------- |

| useSku | SKU数据管理 | SkuListPage |
| useExport | 导出功能 | 多处使用 |
| useLoading | 加载状态管理 | 全局 |

## 依赖关系图

```
HomePage
├── UploadArea
│   └── FileList
├── DetectionList
│   ├── BatchResultCard
│   └── MatchResult
│       └── SkuImage
└── StatusBanner

TaskListPage
├── TaskStats
├── TaskNav
└── ImageViewer

SkuListPage
├── SkuImage
└── ImageViewer

SkuReviewPage
└── ImageViewer
```

## 组件版本管理

### 版本号规则
采用语义化版本控制：`v{major}.{minor}.{patch}`
- major: 重大变更，不兼容升级
- minor: 新增功能，向后兼容
- patch: 修复bug，向后兼容

### 更新记录

| 版本   | 日期       | 更新内容                                     |
| ------ | ---------- | -------------------------------------------- |
| v1.0.0 | 2026-05-23 | 初始版本                                     |
| v1.0.1 | 2026-05-23 | 更新导入路径为别名格式，添加路径别名配置说明 |
| v1.1.0 | 2026-06-02 | 修正组件列表：删除不存在的LoginPage，新增home/目录及组件，补充ui/和task/遗漏组件 |

## 使用指南

### 导入组件

```javascript
// 页面组件
import HomePage from '@pages/HomePage.vue'

// UI组件
import ImageViewer from '@ui/ImageViewer.vue'

// 任务组件
import TaskStats from '@task/TaskStats.vue'

// SKU组件
import SkuImage from '@sku/SkuImage.vue'

// 上传组件
import UploadArea from '@upload/UploadArea.vue'
```

### 组件使用示例

#### ImageViewer
```vue
<ImageViewer 
  :visible="showViewer"
  :image-url="imageUrl"
  :image-name="imageName"
  @update:visible="showViewer = false"
/>
```

#### SkuImage
```vue
<SkuImage 
  :image-path="'/path/to/image.jpg'" 
  :placeholder-icon="'📦'" 
  height="80px" 
/>
```

## 注意事项

1. **组件职责单一**：每个组件只负责一个功能
2. **避免深层嵌套**：组件嵌套不超过3层
3. **props验证**：所有props必须定义类型和默认值
4. **事件命名**：事件名使用 kebab-case
5. **样式隔离**：使用scoped样式避免全局污染
6. **性能优化**：合理使用v-memo、v-lazy等优化渲染
