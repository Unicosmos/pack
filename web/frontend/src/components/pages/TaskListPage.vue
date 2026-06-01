<template>
  <div class="task-list-page">
    <!-- 左右分栏布局 - 不使用PageContainer以获得全屏效果 -->
    <div class="main-layout">
        <!-- 左侧面板 -->
        <div class="left-panel">
          <TaskStats :stats="stats" />

          <!-- 筛选工具栏 -->
          <div class="filter-bar-wrapper">
            <FilterBar>
              <template #filters>
                <FilterDropdown
                  v-model="statusFilter"
                  :options="statusOptions"
                  placeholder="全部状态"
                  @change="handleStatusChange"
                />
                <TimeFilterDropdown
                  v-model="timeFilter"
                  :custom-start="customStart"
                  :custom-end="customEnd"
                  @update:customStart="customStart = $event"
                  @update:customEnd="customEnd = $event"
                  @change="handleTimeChange"
                />
              </template>
              <template #actions>
                <ActionMenu
                  :items="batchActions"
                  :loading="detecting"
                  :progress="detectProgress"
                  :selected-count="selectedTasks.length"
                  @select="handleBatchAction"
                >
                  <template #label>{{ detecting ? '识别中...' : `批量 (${selectedTasks.length})` }}</template>
                </ActionMenu>
              </template>
            </FilterBar>
          </div>

          <!-- 任务列表 -->
          <div class="task-list-wrap">
            <div v-if="loading" class="loading">加载中...</div>

            <div v-else-if="tasks.length === 0" class="list-empty">
              <div class="empty-icon">📭</div>
              <p>暂无任务记录</p>
            </div>

            <div v-else class="task-list-container">
              <!-- 表头 -->
              <div class="task-table-header">
                <div class="th-checkbox">
                  <input type="checkbox" @click.stop="selectAllTasks" :checked="selectedTaskIds.length === tasks.length && tasks.length > 0" />
                </div>
                <div class="th-id">ID</div>
                <div class="th-name">图片名称</div>
                <div class="th-status">状态</div>
                <div class="th-counts">统计</div>
                <div class="th-date">创建时间</div>
              </div>

              <!-- 任务行 -->
              <div
                v-for="task in tasks"
                :key="task.id"
                class="task-row"
                :class="{ 'active': viewDetailTask?.id === task.id }"
                @click="openViewDetail(task)"
              >
                <div class="td-checkbox" @click.stop>
                  <input type="checkbox" :value="task.id" @click.stop="toggleTaskSelection(task.id)" :checked="selectedTaskIds.includes(task.id)" />
                </div>
                <div class="td-id">
                  <div class="task-id-num">#{{ task.id }}</div>
                </div>
                <div class="td-name">
                  <div class="task-name-text">{{ task.image_name }}</div>
                  <div class="task-stat-text">检测:{{ task.box_count || 0 }} · 匹配:{{ task.matched_count || 0 }}</div>
                </div>
                <div class="td-status">
                  <button v-if="shouldShowDetect(task)" class="status-tag pending status-btn" @click.stop="detectTask(task)" title="点击识别">
                    待识别
                  </button>
                  <span v-else :class="['status-tag', getStatusBadgeClass(task)]">{{ getStatusText(task.status) }}</span>
                </div>
                <div class="td-counts">
                  <span class="count-detect">{{ task.box_count || 0 }}</span>
                  <span class="count-sep">/</span>
                  <span class="count-match">{{ task.matched_count || 0 }}</span>
                </div>
                <div class="td-date">{{ formatDateShort(task.created_at) }}</div>
              </div>
            </div>

          </div>
        </div>

        <!-- 右侧面板 - 任务详情 -->
        <div class="right-panel">
          <div v-if="!viewDetailTask" class="detail-empty">
            <div class="empty-icon">📂</div>
            <div>请在左侧选择任务查看详情</div>
            <div class="empty-hint">点击任务开始识别或审核</div>
          </div>

          <template v-else>
            <!-- 详情头部 -->
            <div class="detail-header">
              <div class="detail-header-left">
                <div class="detail-title">
                  #{{ viewDetailTask.id }}
                  <span class="detail-title-name">{{ viewDetailTask.image_name }}</span>
                </div>
                <div class="detail-meta">
                  <span :class="['status-tag', getStatusBadgeClass(viewDetailTask)]">{{ getStatusText(viewDetailTask.status) }}</span>
                  <span class="meta-item">创建: {{ formatDate(viewDetailTask.created_at) }}</span>
                  <span v-if="viewDetailTask.completed_at" class="meta-item">完成: {{ formatDate(viewDetailTask.completed_at) }}</span>
                </div>
              </div>
              <div class="detail-header-right">
                <button v-if="shouldShowDetect(viewDetailTask)" class="btn btn-blue" @click="detectTask(viewDetailTask)">▶ 开始识别</button>
                <button v-else-if="viewDetailTask.status === 'detected'" class="btn btn-orange" @click="startReview">✎ 去审核</button>
                <button v-else class="btn btn-green" @click="exportSingleTask">⬇ 导出结果</button>
              </div>
            </div>

            <!-- 图片对比 -->
            <div class="compare-area">
              <div class="compare-box" @click="openImageViewer(getTaskImagePath(viewDetailTask), viewDetailTask.image_name)">
                <div class="compare-label">
                  <span>原图</span>
                  <span class="compare-link">👆 查看大图</span>
                </div>
                <div class="compare-img-wrapper">
                  <SkuImage
                    :image-path="getTaskImagePath(viewDetailTask)"
                    fit="contain"
                    class="compare-img"
                  />
                </div>
              </div>
              <div v-if="getTaskPreviewPath(viewDetailTask)" class="compare-box" @click="openImageViewer(getTaskPreviewPath(viewDetailTask).url, viewDetailTask.image_name + ' (检测结果)')">
                <div class="compare-label">
                  <span>检测结果（带框）</span>
                  <span class="compare-link">👆 查看大图</span>
                </div>
                <div class="compare-img-wrapper">
                  <SkuImage
                    :image-path="getTaskPreviewPath(viewDetailTask)"
                    fit="contain"
                    class="compare-img"
                  />
                </div>
              </div>
            </div>

            <!-- 箱体结果 -->
            <div v-if="getDetectionBoxes(viewDetailTask).length > 0" class="boxes-section">
              <div class="boxes-header">
                <div class="boxes-title">识别结果 ({{ getDetectionBoxes(viewDetailTask).length }}个箱体)</div>
                <div class="boxes-hint">👆 点击箱体可查看/修改匹配</div>
              </div>
              <div class="boxes-grid">
                <div
                  v-for="(box, idx) in getDetectionBoxes(viewDetailTask)"
                  :key="box.box_id"
                  class="box-card"
                  :class="{ deleted: box.status === 'deleted' }"
                  @click="box.status !== 'deleted' && handleBoxMatch({ box, index: idx })"
                >
                  <div v-if="box.status !== 'deleted'" class="box-card-main">
                    <SkuImage
                      :image-path="getBoxImageUrl(box)"
                      :placeholder-icon="String(idx + 1)"
                      fit="contain"
                      class="box-card-img"
                    />
                    <div class="box-card-info">
                      <div class="box-card-name">箱体 {{ idx + 1 }}</div>
                      <div class="box-card-conf">置信度: {{ (box.confidence * 100).toFixed(1) }}%</div>
                      <div v-if="getMatchResultForTask(viewDetailTask, box.box_id)" class="box-card-match" :class="getMatchResultForTask(viewDetailTask, box.box_id).status">
                        {{ getMatchResultForTask(viewDetailTask, box.box_id).sku_id || '未匹配' }}
                      </div>
                      <div v-else class="box-card-match unmatched">未匹配</div>
                    </div>
                    <button class="btn-delete-box" @click.stop="handleDeleteBox({ box, index: idx })" title="删除此箱体">
                      🗑️
                    </button>
                  </div>
                  <div v-else class="box-card-main box-deleted">
                    <SkuImage
                      :image-path="getBoxImageUrl(box)"
                      :placeholder-icon="String(idx + 1)"
                      fit="contain"
                      class="box-card-img"
                    />
                    <div class="box-card-info">
                      <div class="box-card-name">箱体 {{ idx + 1 }}</div>
                      <div class="box-deleted-text">已删除</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>
    <!-- 匹配弹窗 -->
    <BoxMatchDialog
      :visible="showMatchDialog"
      :box="matchDialogBox"
      :box-index="matchDialogIndex"
      :task-id="viewDetailTask?.id"
      @close="showMatchDialog = false"
      @update="handleBoxMatchUpdate"
      @submit-review="handleSubmitReview"
    />

    <!-- 大图查看器 -->
    <ImageViewer
      :visible="showImageViewer"
      :image-url="currentImageUrl"
      :image-name="currentImageName"
      @update:visible="showImageViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import taskApi from '@api/taskApi'
import { useAppStore } from '@stores/app'
// import PageContainer from '@layout/PageContainer.vue'
import TaskStats from '@task/TaskStats.vue'
import BoxMatchDialog from '@task/BoxMatchDialog.vue'
import ImageViewer from '@ui/ImageViewer.vue'
import FilterBar from '@ui/FilterBar.vue'
import FilterDropdown from '@ui/FilterDropdown.vue'
import TimeFilterDropdown from '@ui/TimeFilterDropdown.vue'
import ActionMenu from '@ui/ActionMenu.vue'
import SkuImage from '@sku/SkuImage.vue'
import { getImageUrlFromPath } from '@api/client'
import {
  getStatusText,
  getStatusBadgeClass,
  formatDate,
  shouldShowDetect,
  shouldShowReDetect
} from '@utils/taskUtils'

const tasks = ref([])
const stats = ref({})
const loading = ref(false)
const submitting = ref(false)
const page = ref(1)
const pageSize = ref(100)  // 后端限制最大100，如需更多需要多次加载
const total = ref(0)
const statusFilter = ref(null)
const timeFilter = ref('all')
const customStart = ref('')
const customEnd = ref('')

const store = useAppStore()

// 筛选选项配置
const statusOptions = [
  { value: null, label: '全部状态' },
  { value: 'completed', label: '已完成' },
  { value: 'pending', label: '待识别' },
  { value: 'detected', label: '待审核' },
  { value: 'failed', label: '识别失败' }
]

// 批量操作选项
const batchActions = [
  { label: '批量识别', icon: '🔍', action: 'detect' },
  { divider: true },
  { label: '导出 JSON', icon: '📄', action: 'export-json' },
  { label: '导出 CSV', icon: '📊', action: 'export-csv' },
  { divider: true },
  { label: '批量删除', icon: '🗑️', action: 'delete', danger: true }
]

// 当前查看的任务详情
const viewDetailTask = ref(null)

// 导出相关
const selectedTaskIds = ref([])
const exporting = ref(false)

// 批量识别加载状态
const detecting = ref(false)
const detectProgress = ref(null)

// 批量删除加载状态
const deleting = ref(false)

// 匹配弹窗相关
const showMatchDialog = ref(false)
const matchDialogBox = ref(null)
const matchDialogIndex = ref(0)

// 大图查看
const showImageViewer = ref(false)
const currentImageUrl = ref('')
const currentImageName = ref('')

const selectedTasks = computed(() => {
  return tasks.value.filter(task => selectedTaskIds.value.includes(task.id))
})

// const totalPages = computed(() => Math.ceil(total.value / pageSize.value) || 1)

const toggleTaskSelection = (taskId) => {
  const index = selectedTaskIds.value.indexOf(taskId)
  if (index === -1) {
    selectedTaskIds.value.push(taskId)
  } else {
    selectedTaskIds.value.splice(index, 1)
  }
}

const selectAllTasks = () => {
  if (selectedTaskIds.value.length === tasks.value.length) {
    selectedTaskIds.value = []
  } else {
    selectedTaskIds.value = tasks.value.map(t => t.id)
  }
}

const getTaskImagePath = (task) => {
  return `/api/tasks/${task.id}/image`
}

const getTaskPreviewPath = (task) => {
  if (task.id && (task.status === 'detected' || task.status === 'completed')) {
    return { url: `/api/tasks/${task.id}/detection-image` }
  }
  return null
}

const getDetectionBoxes = (task) => {
  if (!task) return []
  if (task.detections) return task.detections
  if (task.result?.detections?.boxes) return task.result.detections.boxes
  return []
}

const getBoxImageUrl = (box) => {
  if (!box) return ''
  if (box.crop_base64) {
    return { url: 'data:image/jpeg;base64,' + box.crop_base64 }
  }
  if (box.crop_path) {
    return getImageUrlFromPath(box.crop_path)
  }
  return ''
}

const getMatchResultForTask = (task, boxId) => {
  const boxes = getDetectionBoxes(task)
  if (boxes.length === 0) return null
  const box = boxes.find(b => b.box_id === boxId || b.box_id === `box_${boxId}` || String(b.box_id) === String(boxId))
  return box ? box.match_result : null
}

const formatDateShort = (dateStr) => {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const isToday = date.toDateString() === now.toDateString()
  if (isToday) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

const openViewDetail = async (task) => {
  try {
    const res = await taskApi.getTaskDetections(task.id)
    if (res.success && res.data && res.data.boxes) {
      viewDetailTask.value = {
        ...task,
        result: {
          detections: {
            boxes: res.data.boxes
          }
        }
      }
    } else {
      viewDetailTask.value = task
    }
  } catch (e) {
    viewDetailTask.value = task
  }
}

const handleBoxMatch = ({ box, index }) => {
  matchDialogBox.value = box
  matchDialogIndex.value = index
  showMatchDialog.value = true
}

const handleBoxMatchUpdate = ({ boxId, skuId, boxIndex }) => {
  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes) return

  const box = boxes.find(b => b.box_id === boxId || b.box_id === `box_${boxId}` || String(b.box_id) === String(boxId))
  if (!box) return

  if (box.match_result) {
    box.match_result.sku_id = skuId
    box.match_result.status = 'matched'
  } else {
    box.match_result = { sku_id: skuId, status: 'matched' }
  }

  box.isModified = true

  ElMessage.success(`箱体 ${boxIndex + 1} 已更新为 ${skuId}`)
}

const handleDeleteBox = async ({ box, index }) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除箱体 ${index + 1} 吗？`,
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
  } catch {
    return
  }

  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes) return

  const targetBox = boxes.find(b => b.box_id === box.box_id || b.box_id === `box_${box.box_id}` || String(b.box_id) === String(box.box_id))
  if (!targetBox) return

  targetBox.status = 'deleted'
  ElMessage.success(`箱体 ${index + 1} 已删除`)
}

const handleSubmitReview = async () => {
  if (!viewDetailTask.value) return

  const boxes = viewDetailTask.value.detections || viewDetailTask.value.result?.detections?.boxes
  if (!boxes || boxes.length === 0) {
    ElMessage.warning('没有检测结果可提交')
    return
  }

  const activeBoxes = boxes.filter(b => b.status !== 'deleted')
  if (activeBoxes.length === 0) {
    ElMessage.warning('所有箱体都已删除，请恢复至少一个箱体后再提交')
    return
  }

  try {
    const resultBoxes = activeBoxes.map((b, idx) => ({
      box_id: `box_${idx}`,
      status: b.status === 'deleted' ? 'deleted' : (b.status || 'approved'),
      is_manual_override: b.isModified || false,
      custom_sku: b.match_result?.sku_id
    }))

    const res = await taskApi.reviewTask(viewDetailTask.value.id, resultBoxes)
    if (res.success) {
      ElMessage.success(res.message || '审核提交成功')
      viewDetailTask.value.status = 'completed'
      await loadTasks()
      await loadStats()
    } else {
      ElMessage.error('提交审核失败')
    }
  } catch (e) {
    ElMessage.error('提交审核失败: ' + (e.detail || e.message || '未知错误'))
  }
}

const startReview = () => {
  // 触发审核模式，可以展开第一个未匹配的箱体
  const boxes = getDetectionBoxes(viewDetailTask.value)
  const firstUnmatched = boxes.find((b, idx) => {
    const match = getMatchResultForTask(viewDetailTask.value, b.box_id)
    return b.status !== 'deleted' && (!match || match.status !== 'matched')
  })
  if (firstUnmatched) {
    const idx = boxes.indexOf(firstUnmatched)
    handleBoxMatch({ box: firstUnmatched, index: idx })
  }
}

const exportSingleTask = async () => {
  if (!viewDetailTask.value) return
  selectedTaskIds.value = [viewDetailTask.value.id]
  await exportTasks('json', false)
  selectedTaskIds.value = []
}

const detectTask = async (task) => {
  submitting.value = true
  try {
    const res = await taskApi.detectTask(task.id)
    if (res.success && res.data && res.data.status === 'detected') {
      ElMessage.success('检测成功')
      await loadTasks()
      await loadStats()
      // 如果当前正在查看此任务，刷新详情
      if (viewDetailTask.value?.id === task.id) {
        await openViewDetail(task)
      }
    } else {
      ElMessage.error('检测未成功执行')
    }
  } catch (e) {
    ElMessage.error('检测失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    submitting.value = false
  }
}

const loadTasks = async () => {
  loading.value = true
  try {
    const customTime = timeFilter.value === 'custom'
    let startUtc = null
    let endUtc = null
    if (customTime && customStart.value) {
      startUtc = new Date(customStart.value).toISOString().slice(0, 16)
    }
    if (customTime && customEnd.value) {
      endUtc = new Date(customEnd.value).toISOString().slice(0, 16)
    }
    const res = await taskApi.listTasks({
      page: page.value,
      page_size: pageSize.value,
      status_filter: statusFilter.value,
      time_filter: customTime ? null : timeFilter.value,
      start_time: startUtc,
      end_time: endUtc
    })
    if (res.success) {
      tasks.value = res.data.tasks
      total.value = res.data.total

      // 如果当前正在查看详情，同步更新基本信息
      if (viewDetailTask.value) {
        const updatedTask = tasks.value.find(t => t.id === viewDetailTask.value.id)
        if (updatedTask) {
          viewDetailTask.value = {
            ...updatedTask,
            detections: viewDetailTask.value.detections,
            result: viewDetailTask.value.result
          }
        }
      }
    }
  } catch (e) {
    ElMessage.error('加载任务列表失败: ' + (e.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

const loadStats = async () => {
  try {
    const customTime = timeFilter.value === 'custom'
    let startUtc = null
    let endUtc = null
    if (customTime && customStart.value) {
      startUtc = new Date(customStart.value).toISOString().slice(0, 16)
    }
    if (customTime && customEnd.value) {
      endUtc = new Date(customEnd.value).toISOString().slice(0, 16)
    }
    const res = await taskApi.getTaskStats(
      customTime ? null : timeFilter.value,
      startUtc,
      endUtc,
      statusFilter.value
    )
    if (res.success) {
      stats.value = res.data
    }
  } catch (e) {
    console.error('加载统计失败', e)
  }
}

const handleStatusChange = () => {
  page.value = 1
  loadTasks()
  loadStats()
}

const handleTimeChange = (value) => {
  if (value && value.type === 'custom') {
    if (!customStart.value || !customEnd.value) return
    if (new Date(customEnd.value) <= new Date(customStart.value)) {
      ElMessage.warning('结束时间必须晚于开始时间')
      return
    }
  }
  page.value = 1
  loadTasks()
  loadStats()
}

const handleBatchAction = (action) => {
  if (action === 'detect') {
    batchDetectTasks()
  } else if (action === 'export-json') {
    exportTasks('json', false)
  } else if (action === 'export-csv') {
    exportTasks('csv', false)
  } else if (action === 'delete') {
    batchDeleteTasks()
  }
}

// 分页功能已移除，保留此函数以防其他代码引用
const changePage = (newPage) => {
  console.log('分页功能已移除')
}

const batchDetectTasks = async () => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要识别的任务')
    return
  }

  detecting.value = true
  detectProgress.value = 0

  try {
    const total = selectedTasks.value.length
    let completed = 0

    for (const task of selectedTasks.value) {
      if (shouldShowDetect(task) || shouldShowReDetect(task)) {
        await taskApi.detectTask(task.id)
      }
      completed++
      detectProgress.value = Math.round((completed / total) * 100)
    }

    ElMessage.success('批量识别完成')
    selectedTaskIds.value = []
    await loadTasks()
    await loadStats()
  } catch (e) {
    ElMessage.error('批量识别失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    detecting.value = false
    detectProgress.value = null
  }
}

const batchDeleteTasks = async () => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要删除的任务')
    return
  }

  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${selectedTasks.value.length} 个任务吗？此操作不可恢复。`,
      '确认批量删除',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
  } catch {
    return
  }

  deleting.value = true
  try {
    const taskIds = selectedTasks.value.map(task => task.id)
    const res = await taskApi.batchDeleteTasks(taskIds)
    if (res.success) {
      ElMessage.success(res.data.message || `成功删除 ${selectedTasks.value.length} 个任务`)
      selectedTaskIds.value = []
      // 如果删除的是当前查看的任务，清空详情
      if (viewDetailTask.value && taskIds.includes(viewDetailTask.value.id)) {
        viewDetailTask.value = null
      }
      await loadTasks()
      await loadStats()
    } else {
      ElMessage.error('批量删除失败')
    }
  } catch (e) {
    ElMessage.error('批量删除失败: ' + (e.detail || e.message || '未知错误'))
  } finally {
    deleting.value = false
  }
}

const openImageViewer = (imageUrl, imageName = '图片') => {
  currentImageUrl.value = imageUrl
  currentImageName.value = imageName
  showImageViewer.value = true
}

const exportTasks = async (format, includeImages) => {
  if (selectedTasks.value.length === 0) {
    ElMessage.warning('请先选择要导出的任务')
    return
  }

  exporting.value = true

  const taskIds = selectedTasks.value.map(task => task.id)

  try {
    const loadingInstance = ElLoading.service({
      lock: true,
      text: `正在导出 ${selectedTasks.value.length} 个任务...`,
      background: 'rgba(0, 0, 0, 0.7)'
    })

    const response = await taskApi.batchExportTasks(taskIds, format)

    if (!response.success) {
      throw new Error(response.error || '导出请求失败')
    }

    const blob = response.data
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url

    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
    const filename = `batch_export_${selectedTasks.value.length}_tasks_${timestamp}.${format === 'json' ? 'json' : 'csv'}`
    a.download = filename

    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)

    loadingInstance.close()

    ElMessage({
      message: `成功导出 ${selectedTasks.value.length} 个任务！`,
      type: 'success',
      duration: 3000
    })
  } catch (e) {
    console.error('导出失败:', e)
    ElMessage({
      message: '导出失败: ' + (e.message || '未知错误'),
      type: 'error',
      duration: 5000
    })
  } finally {
    exporting.value = false
  }
}

const openPendingTask = () => {
  const pendingId = store.pendingTaskId
  if (pendingId === null) return

  store.pendingTaskId = null

  const task = tasks.value.find(t => t.id === pendingId)
  if (task) {
    openViewDetail(task)
  } else {
    taskApi.getTask(pendingId).then(res => {
      if (res.success && res.data) {
        openViewDetail(res.data)
      }
    })
  }
}

onMounted(async () => {
  await loadTasks()
  await loadStats()
  openPendingTask()
})
</script>

<style scoped>
.task-list-page {
  height: 100%;
  overflow: hidden;
}

/* 左右分栏布局 */
.main-layout {
  display: flex;
  height: 100%;
  overflow: hidden;
}

/* 左侧面板 */
.left-panel {
  width: 420px;
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  background: var(--color-bg-secondary);
  overflow: hidden;
  flex-shrink: 0;
}

.dark .left-panel {
  background: rgba(30, 41, 59, 0.4);
}

/* 筛选工具栏 */
.filter-bar-wrapper {
  padding: 10px 12px;
  flex-shrink: 0;
  position: relative;
}

/* 任务列表区域 */
.task-list-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 0 12px 8px;
  position: relative;
}

/* 表头 */
.task-table-header {
  display: grid;
  grid-template-columns: 28px 50px 1fr 70px 60px 90px;
  gap: 6px;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-border);
  font-size: 11px;
  color: var(--color-text-tertiary);
  font-weight: 500;
  position: sticky;
  top: 0;
  background: var(--color-bg-tertiary);
  z-index: 10;
}

.dark .task-table-header {
  background: rgba(30, 41, 59, 0.95);
}

/* 任务行 */
.task-row {
  display: grid;
  grid-template-columns: 28px 50px 1fr 70px 60px 90px;
  gap: 6px;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-border-light);
  cursor: pointer;
  transition: all 0.15s;
  border-radius: 6px;
  padding-left: 6px;
  padding-right: 6px;
  margin-left: -6px;
  margin-right: -6px;
}

.task-row:hover {
  background: var(--color-bg-hover);
}

.task-row.active {
  background: rgba(0, 102, 204, 0.08);
  border-left: 3px solid var(--color-primary);
  margin-left: -9px;
  padding-left: 8px;
}

.dark .task-row {
  border-bottom-color: rgba(51, 65, 85, 0.3);
}

.dark .task-row:hover {
  background: rgba(255, 255, 255, 0.03);
}

.dark .task-row.active {
  background: rgba(0, 102, 204, 0.15);
}

.task-row.active .task-name-text {
  color: var(--color-primary);
  font-weight: 500;
}

/* 单元格 */
.th-checkbox, .td-checkbox {
  display: flex;
  align-items: center;
  justify-content: center;
}

.td-checkbox input, .th-checkbox input {
  width: 14px;
  height: 14px;
  accent-color: var(--color-primary);
  cursor: pointer;
}

.th-id, .td-id {
  display: flex;
  align-items: center;
}

.task-id-num {
  font-size: 11px;
  color: var(--color-text-tertiary);
  font-weight: 500;
}

.th-name, .td-name {
  min-width: 0;
}

.task-name-text {
  font-size: 12px;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.task-stat-text {
  font-size: 10px;
  color: var(--color-text-tertiary);
  margin-top: 1px;
}

.th-status, .td-status {
  display: flex;
  align-items: center;
}

.status-tag {
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  display: inline-block;
  white-space: nowrap;
}

.status-tag.pending {
  background: rgba(245, 158, 11, 0.1);
  color: var(--color-warning);
}

.status-tag.detected {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.status-tag.completed {
  background: rgba(34, 197, 94, 0.1);
  color: var(--color-success);
}

.status-tag.failed {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
}

.status-tag.unmatched {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
}

.status-btn {
  border: none;
  cursor: pointer;
  font: inherit;
  outline: none;
  transition: opacity 0.15s;
}

.status-btn:hover {
  opacity: 0.8;
}

.th-counts, .td-counts {
  display: flex;
  align-items: center;
  gap: 2px;
  font-size: 11px;
  color: var(--color-text-secondary);
}

.count-detect {
  color: var(--color-text-secondary);
}

.count-sep {
  color: var(--color-text-tertiary);
}

.count-match {
  color: var(--color-success);
  font-weight: 500;
}

.th-date, .td-date {
  font-size: 11px;
  color: var(--color-text-tertiary);
}

/* 空状态 */
.list-empty {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-text-tertiary);
  font-size: 13px;
}

.list-empty .empty-icon {
  font-size: 48px;
  margin-bottom: var(--spacing-md);
}

.loading {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--color-text-secondary);
}



/* 右侧面板 */
.right-panel {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: var(--color-bg-primary);
}

/* 空状态 */
.detail-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  gap: 12px;
}

.detail-empty .empty-icon {
  font-size: 48px;
}

.detail-empty .empty-hint {
  font-size: 12px;
  color: var(--color-text-tertiary);
  margin-top: 4px;
}

/* 详情头部 */
.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-border);
}

.detail-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.detail-title-name {
  color: var(--color-text-secondary);
  font-weight: 400;
  margin-left: 8px;
  font-size: 15px;
}

.detail-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.meta-item {
  color: var(--color-text-secondary);
}

.detail-header-right {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 8px 16px;
  border-radius: 6px;
  border: none;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.btn-blue {
  background: var(--color-primary);
  color: #fff;
}

.btn-blue:hover {
  background: var(--color-primary-dark);
}

.btn-orange {
  background: rgba(245, 158, 11, 0.15);
  color: var(--color-warning);
  border: 1px solid rgba(245, 158, 11, 0.3);
}

.btn-orange:hover {
  background: rgba(245, 158, 11, 0.25);
}

.btn-green {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.btn-green:hover {
  background: rgba(34, 197, 94, 0.25);
}

/* 图片对比 */
.compare-area {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

.compare-box {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: border-color 0.2s;
}

.compare-box:hover {
  border-color: var(--color-primary);
}

.compare-label {
  padding: 10px 14px;
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
  color: var(--color-text-secondary);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.compare-link {
  color: var(--color-primary);
  font-size: 12px;
}

.compare-img-wrapper {
  width: 100%;
  height: 280px;
  background: var(--color-bg-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.dark .compare-img-wrapper {
  background: #0f172a;
}

.compare-img-wrapper :deep(.sku-image) {
  height: 100% !important;
  width: 100%;
}

.compare-img-wrapper :deep(.sku-image img) {
  object-fit: contain;
  max-height: 280px;
}

/* 箱体结果 */
.boxes-section {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
}

.boxes-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.boxes-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.boxes-hint {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.boxes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.box-card {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.dark .box-card {
  background: #0f172a;
}

.box-card:hover {
  border-color: var(--color-primary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.box-card.deleted {
  opacity: 0.5;
  cursor: default;
}

.box-card.deleted:hover {
  border-color: var(--color-border);
  transform: none;
  box-shadow: none;
}

.box-card-main {
  position: relative;
}

.box-card-img {
  width: 100%;
  height: 130px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-tertiary);
}

.dark .box-card-img {
  background: #1e293b;
}

.box-card-img :deep(.sku-image) {
  height: 100% !important;
  width: 100%;
}

.box-card-img :deep(.sku-image img) {
  object-fit: contain;
  max-height: 130px;
}

.box-card-info {
  padding: 10px;
}

.box-card-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.box-card-conf {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-bottom: 6px;
}

.box-card-match {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  text-align: center;
}

.box-card-match.matched {
  background: rgba(34, 197, 94, 0.1);
  color: var(--color-success);
}

.box-card-match.unmatched {
  background: rgba(239, 68, 68, 0.1);
  color: var(--color-danger);
}

.box-deleted-text {
  color: var(--color-text-secondary);
  font-size: 12px;
}

.btn-delete-box {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 24px;
  height: 24px;
  border: none;
  background: rgba(245, 108, 108, 0.9);
  color: white;
  border-radius: 50%;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  z-index: 10;
  opacity: 0;
}

.box-card:hover .btn-delete-box {
  opacity: 1;
}

.btn-delete-box:hover {
  background: rgba(220, 53, 69, 1);
  transform: scale(1.1);
}

/* 响应式 */
@media (max-width: 900px) {
  .left-panel {
    width: 320px;
  }

  .compare-area {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .main-layout {
    flex-direction: column;
  }

  .left-panel {
    width: 100%;
    height: 50%;
    border-right: none;
    border-bottom: 1px solid var(--color-border);
  }

  .right-panel {
    height: 50%;
  }
}


</style>
