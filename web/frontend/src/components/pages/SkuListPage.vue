<template>
  <div class="sku-list-page">
    <!-- 主内容区 -->
    <div class="main">
      <!-- 工具栏 -->
      <div class="toolbar">
        <div class="search-box">
          <input
            v-model="searchQuery"
            type="text"
            placeholder="搜索SKU编号或名称..."
            @keyup.enter="doSearch"
          />
        </div>

        <div class="view-switch">
          <button :class="{ active: viewMode === 'list' }" @click="viewMode = 'list'">
            <span>☰</span> 列表
          </button>
          <button :class="{ active: viewMode === 'gallery' }" @click="viewMode = 'gallery'">
            <span>▦</span> 画廊
          </button>
        </div>

        <div class="toolbar-actions">
          <button class="btn btn-secondary" @click="handleSyncFromCsv">🔄 同步CSV</button>
          <button class="btn btn-blue" @click="handleExport">📥 导出全部</button>
        </div>
      </div>

      <!-- 统计摘要栏 -->
      <div class="stats-summary">
        <span>📦 共 <b>{{ displayStats.total || 0 }}</b> 个SKU</span>
        <span>🖼️ <b>{{ displayStats.images || 0 }}</b> 张图片</span>
      </div>

      <!-- 内容区 -->
      <div class="content-wrap">
        <!-- 批量操作栏 -->
        <div class="batch-bar" v-if="selectedSkus.length > 0">
          <span>已选择 <b>{{ selectedSkus.length }}</b> 个SKU</span>
          <button class="btn btn-red" @click="batchDelete">🗑️ 批量删除</button>
          <button class="btn btn-blue" @click="batchExport">📥 批量导出</button>
          <button class="btn btn-secondary" @click="clearSelection">清空</button>
        </div>

        <div v-if="loading" class="loading">加载中...</div>

        <div v-else-if="skus.length === 0" class="empty-state">
          <div class="empty-icon">📭</div>
          <p>暂无SKU数据</p>
          <p>请通过同步CSV导入SKU数据</p>
        </div>

        <!-- 列表视图 -->
        <table v-else-if="viewMode === 'list'" class="sku-table">
          <thead>
            <tr>
              <th class="chk-cell">
                <input type="checkbox" v-model="selectAll" @change="toggleSelectAll" />
              </th>
              <th class="id-cell">SKU编号</th>
              <th class="thumb-cell">缩略图</th>
              <th>名称</th>
              <th class="count-cell">图片数</th>
              <th class="action-cell">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="sku in skus" 
              :key="sku.id" 
              :class="{ selected: selectedSku?.sku_id === sku.sku_id }"
              @click="selectSku(sku)"
            >
              <td class="chk-cell" @click.stop>
                <input type="checkbox" :value="sku.sku_id" v-model="selectedSkus" />
              </td>
              <td class="id-cell">{{ sku.sku_id }}</td>
              <td class="thumb-cell">
                <div class="sku-thumb">
                  <SkuImage :image-path="getFirstImagePath(sku.sku_id)" width="100%" height="100%" />
                </div>
              </td>
              <td>{{ sku.sku_name }}</td>
              <td class="count-cell">{{ sku.image_count }}</td>
              <td class="action-cell">
                <div class="action-btns">
                  <button class="icon-btn edit" @click.stop="openEditDialog(sku)" title="编辑">✎</button>
                  <button class="icon-btn delete" @click.stop="confirmDelete(sku)" title="删除">🗑️</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>

        <!-- 画廊视图 -->
        <div v-else class="gallery-view">
          <div 
            v-for="sku in skus" 
            :key="sku.id" 
            :class="['gallery-card', { selected: selectedSku?.sku_id === sku.sku_id }]"
            @click="selectSku(sku)"
          >
            <div class="chk-wrap" @click.stop>
              <input type="checkbox" :value="sku.sku_id" v-model="selectedSkus" />
            </div>
            <div class="gallery-image-box">
              <SkuImage :image-path="getFirstImagePath(sku.sku_id)" width="100%" height="100%" />
              <div v-if="sku.image_count > 0" class="img-count">{{ sku.image_count }}张</div>
            </div>
            <div class="gallery-info">
              <div class="gallery-sku">{{ sku.sku_id }}</div>
              <div class="gallery-name">{{ sku.sku_name }}</div>
            </div>
          </div>
        </div>
      </div>
      <!-- 分页 -->
      <div v-if="total > pageSize" class="pagination-bar">
        <span class="pagination-info">共 {{ total }} 条</span>
        <div class="pagination-controls">
          <button class="page-btn" :disabled="page <= 1" @click="changePage(page - 1)">«</button>
          <template v-for="p in totalPages" :key="p">
            <button
              v-if="p === 1 || p === totalPages || Math.abs(p - page) <= 2"
              :class="['page-btn', { active: p === page }]"
              @click="changePage(p)"
            >{{ p }}</button>
            <span v-else-if="p === page - 3 || p === page + 3" class="page-ellipsis">…</span>
          </template>
          <button class="page-btn" :disabled="page >= totalPages" @click="changePage(page + 1)">»</button>
        </div>
      </div>
    </div>

    <!-- 遮罩 -->
    <div class="overlay" :class="{ show: selectedSku }" @click="closeDetail"></div>

    <!-- 详情面板 -->
    <div class="detail-panel" :class="{ show: selectedSku }">
      <div class="detail-panel-header">
        <div class="detail-panel-title">{{ selectedSku?.sku_id }} - {{ selectedSku?.sku_name }}</div>
        <button class="detail-panel-close" @click="closeDetail">✕</button>
      </div>
      <div class="detail-panel-body" v-if="selectedSku">
        <div>
          <div class="detail-info-grid">
            <div class="info-card">
              <div class="label">分类</div>
              <div class="value">{{ selectedSku.category || '-' }}</div>
            </div>
            <div class="info-card">
              <div class="label">图片数量</div>
              <div class="value">{{ selectedSku.image_count }}</div>
            </div>
            <div class="info-card">
              <div class="label">创建时间</div>
              <div class="value">{{ formatDate(selectedSku.created_at) }}</div>
            </div>
            <div class="info-card info-full">
              <div class="label">描述</div>
              <div class="value">{{ selectedSku.description || '-' }}</div>
            </div>
          </div>
        </div>
        <div>
          <div class="img-manage-title">🖼️ 图片管理（{{ currentSkuImages.length }}张）</div>
          <div class="img-grid">
            <div 
              v-for="(img, idx) in currentSkuImages" 
              :key="idx"
              class="img-item"
              @click="openImageViewer(img.url, img.filename)"
            >
              <SkuImage :image-path="img" width="100%" height="100%" />
              <button class="img-del" @click.stop="deleteImage(idx)">✕</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 编辑对话框 -->
    <div v-if="showEditDialog" class="modal-overlay" @click.self="closeDialog">
      <div class="modal">
        <div class="modal-header">
          <h3>编辑SKU</h3>
          <button class="btn-close" @click="closeDialog">✕</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>SKU编号 <span class="required">*</span></label>
            <input v-model="formData.sku_id" type="text" disabled />
          </div>
          <div class="form-group">
            <label>SKU名称 <span class="required">*</span></label>
            <input v-model="formData.sku_name" type="text" placeholder="请输入SKU名称" />
          </div>
          <div class="form-group">
            <label>分类</label>
            <input v-model="formData.category" type="text" placeholder="请输入分类" />
          </div>
          <div class="form-group">
            <label>描述</label>
            <textarea v-model="formData.description" placeholder="请输入描述" rows="3"></textarea>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="closeDialog">取消</button>
          <button class="btn btn-primary" @click="handleSubmit" :disabled="submitting">
            {{ submitting ? '提交中...' : '确定' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 确认删除对话框 -->
    <div v-if="showDeleteDialog" class="modal-overlay" @click.self="showDeleteDialog = false">
      <div class="modal modal-sm">
        <div class="modal-header">
          <h3>确认删除</h3>
          <button class="btn-close" @click="showDeleteDialog = false">✕</button>
        </div>
        <div class="modal-body">
          <p v-if="deleteType === 'single'">
            确定要永久删除SKU <strong>{{ deleteTarget?.sku_id }}</strong> 吗？<br/>
            <span style="font-size:12px;color:var(--color-text-tertiary)">此操作将删除数据库记录、图片文件和特征数据，不可恢复。</span>
          </p>
          <p v-else>
            确定要永久删除选中的 <strong>{{ selectedSkus.length }}</strong> 个SKU吗？<br/>
            <span style="font-size:12px;color:var(--color-text-tertiary)">此操作不可恢复。</span>
          </p>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showDeleteDialog = false">取消</button>
          <button class="btn btn-danger" @click="handleDelete" :disabled="submitting">
            {{ submitting ? '删除中...' : '确认删除' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Toast -->
    <div v-if="toast.show" :class="['toast', toast.type]">{{ toast.message }}</div>

    <!-- 图片查看器 -->
    <ImageViewer
      :visible="showImageViewer"
      :image-url="viewerImageUrl"
      :image-name="viewerImageName"
      @update:visible="showImageViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { sku } from '@api/client'
import SkuImage from '@sku/SkuImage.vue'
import ImageViewer from '@ui/ImageViewer.vue'

const viewMode = ref('list')
const skus = ref([])
const stats = ref({})
const loading = ref(false)
const submitting = ref(false)

const page = ref(1)
const pageSize = ref(20)
const total = ref(0)
const totalPages = computed(() => Math.ceil(total.value / pageSize.value) || 1)

const searchQuery = ref('')
const categoryFilter = ref('')

const selectedSkus = ref([])
const selectAll = ref(false)
const selectedSku = ref(null)
const currentSkuImages = ref([])

const showImageViewer = ref(false)
const viewerImageUrl = ref('')
const viewerImageName = ref('')

const showEditDialog = ref(false)
const showDeleteDialog = ref(false)

const deleteType = ref('single')
const deleteTarget = ref(null)

const formData = ref({
  sku_id: '',
  sku_name: '',
  description: '',
  category: '',
  status: 'active'
})

const toast = ref({ show: false, message: '', type: 'info' })

const skuImagesCache = ref({})

const displayStats = computed(() => {
  if (stats.value && Object.keys(stats.value).length > 0) {
    return {
      total: stats.value.total_skus || 0,
      images: stats.value.total_images || 0
    }
  }
  return {
    total: skus.value.length,
    images: skus.value.reduce((sum, s) => sum + (s.image_count || 0), 0)
  }
})

const showToast = (message, type = 'info') => {
  toast.value = { show: true, message, type }
  setTimeout(() => {
    toast.value.show = false
  }, 3000)
}

const formatDate = (dateStr) => {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}

const getSkuImages = (skuId) => {
  return skuImagesCache.value[skuId] || []
}

const getFirstImagePath = (skuId) => {
  const images = getSkuImages(skuId)
  return images.length > 0 ? images[0] : ''
}

const loadSkus = async () => {
  loading.value = true
  try {
    const res = await sku.list(page.value, pageSize.value, searchQuery.value, categoryFilter.value)
    if (res.success) {
      skus.value = res.skus || []
      total.value = res.total || 0
      await preloadSkuImages()
    }
  } catch (err) {
    showToast('加载失败: ' + err.message, 'error')
  } finally {
    loading.value = false
  }
}

const preloadSkuImages = async () => {
  for (const s of skus.value) {
    if (!skuImagesCache.value[s.sku_id]) {
      try {
        const res = await sku.getImages(s.sku_id)
        if (res.success) {
          skuImagesCache.value[s.sku_id] = res.images || []
        }
      } catch (err) {
        skuImagesCache.value[s.sku_id] = []
      }
    }
  }
}

const loadStats = async () => {
  try {
    const res = await sku.stats()
    if (res.success) {
      stats.value = res
    }
  } catch (err) {
    console.error('Failed to load stats:', err)
  }
}

const doSearch = () => {
  page.value = 1
  loadSkus()
}

const changePage = (newPage) => {
  if (newPage < 1 || newPage > totalPages.value) return
  page.value = newPage
  loadSkus()
}

const selectSku = async (skuItem) => {
  selectedSku.value = skuItem
  await loadSkuImages(skuItem.sku_id)
}

const loadSkuImages = async (skuId) => {
  try {
    const res = await sku.getImages(skuId)
    if (res.success) {
      currentSkuImages.value = res.images || []
    }
  } catch (err) {
    currentSkuImages.value = []
  }
}

const closeDetail = () => {
  selectedSku.value = null
}

const openEditDialog = (item) => {
  formData.value = {
    sku_id: item.sku_id,
    sku_name: item.sku_name,
    description: item.description || '',
    category: item.category || '',
    status: item.status
  }
  showEditDialog.value = true
}

const closeDialog = () => {
  showEditDialog.value = false
}

const handleSubmit = async () => {
  if (!formData.value.sku_name.trim()) {
    showToast('SKU名称不能为空', 'error')
    return
  }

  submitting.value = true
  try {
    const res = await sku.update(formData.value.sku_id, formData.value)
    if (res.sku_id || res.success) {
      showToast('更新成功')
      closeDialog()
      loadSkus()
      loadStats()
      loadCategories()
    } else {
      showToast(res.detail || '操作失败', 'error')
    }
  } catch (err) {
    showToast('操作失败: ' + (err.detail || err.message), 'error')
  } finally {
    submitting.value = false
  }
}

const confirmDelete = (item) => {
  deleteType.value = 'single'
  deleteTarget.value = item
  showDeleteDialog.value = true
}

const handleDelete = async () => {
  submitting.value = true
  try {
    if (deleteType.value === 'single') {
      const res = await sku.delete(deleteTarget.value.sku_id)
      if (!res.success) throw new Error(res.detail || '删除失败')
    } else {
      const res = await sku.batchDelete(selectedSkus.value)
      if (!res.success) throw new Error(res.detail || '批量删除失败')
    }
    showToast('删除成功')
    showDeleteDialog.value = false
    clearSelection()
    page.value = 1
    loadSkus()
    loadStats()
  } catch (err) {
    showToast('操作失败: ' + (err.detail || err.message), 'error')
  } finally {
    submitting.value = false
  }
}

const toggleSelectAll = () => {
  if (selectAll.value) {
    selectedSkus.value = skus.value.map(s => s.sku_id)
  } else {
    selectedSkus.value = []
  }
}

watch(selectedSkus, (newVal) => {
  if (skus.value.length > 0 && newVal.length === skus.value.length) {
    selectAll.value = true
  } else {
    selectAll.value = false
  }
})

const clearSelection = () => {
  selectedSkus.value = []
  selectAll.value = false
}

const batchDelete = () => {
  deleteType.value = 'batch'
  showDeleteDialog.value = true
}

const batchExport = async () => {
  try {
    await sku.exportSelectedCsv(selectedSkus.value)
    showToast('批量导出成功')
    clearSelection()
  } catch (err) {
    showToast('导出失败: ' + err.message, 'error')
  }
}

const handleSyncFromCsv = async () => {
  try {
    const res = await sku.syncFromCsv()
    if (res.success) {
      showToast(res.message || '同步成功')
      loadSkus()
      loadStats()
      loadCategories()
    } else {
      showToast(res.detail || '同步失败', 'error')
    }
  } catch (err) {
    showToast('同步失败: ' + err.message, 'error')
  }
}

const handleExport = async () => {
  try {
    await sku.exportCsv()
    showToast('导出成功')
  } catch (err) {
    showToast('导出失败: ' + err.message, 'error')
  }
}

const deleteImage = async (index) => {
  if (!selectedSku.value) return
  
  try {
    const img = currentSkuImages.value[index]
    const res = await sku.deleteImage(selectedSku.value.sku_id, img.filename)
    if (res.success) {
      showToast('删除成功')
      await loadSkuImages(selectedSku.value.sku_id)
      loadSkus()
      loadStats()
    } else {
      showToast(res.detail || '删除失败', 'error')
    }
  } catch (err) {
    showToast('删除失败: ' + err.message, 'error')
  }
}

const addImage = () => {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = 'image/*'
  input.multiple = true
  input.onchange = async (e) => {
    const files = Array.from(e.target.files)
    if (files.length === 0) return
    
    submitting.value = true
    try {
      for (const file of files) {
        await sku.uploadImage(selectedSku.value.sku_id, file)
      }
      showToast(`成功上传 ${files.length} 张图片`)
      await loadSkuImages(selectedSku.value.sku_id)
      loadSkus()
      loadStats()
    } catch (err) {
      showToast('上传失败: ' + err.message, 'error')
    } finally {
      submitting.value = false
    }
  }
  input.click()
}

const openImageViewer = (imageUrl, imageName = '图片') => {
  viewerImageUrl.value = imageUrl
  viewerImageName.value = imageName
  showImageViewer.value = true
}

onMounted(() => {
  loadSkus()
  loadStats()
})
</script>

<style scoped>
.sku-list-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-secondary, #0b1120);
  overflow: hidden;
}

.sku-list-page .main {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

/* 工具栏 */
.toolbar {
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  background: var(--color-bg-tertiary);
}

.dark .toolbar {
  background: rgba(30, 41, 59, 0.4);
}

.search-box {
  flex: 1;
  max-width: 300px;
  position: relative;
}

.search-box::before {
  content: "🔍";
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 12px;
  opacity: 0.5;
}

.search-box input {
  width: 100%;
  padding: 8px 12px 8px 36px;
  background: var(--color-bg-tertiary, #0f172a);
  border: 1px solid var(--color-border, #334155);
  border-radius: 8px;
  color: var(--color-text-primary, #e2e8f0);
  font-size: 13px;
  outline: none;
}

.search-box input:focus {
  border-color: var(--color-primary, #3b82f6);
}

.view-switch {
  display: flex;
  background: var(--color-bg-tertiary, #0f172a);
  border: 1px solid var(--color-border, #334155);
  border-radius: 8px;
  overflow: hidden;
}

.view-switch button {
  padding: 8px 16px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary, #94a3b8);
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s;
}

.view-switch button.active {
  background: var(--color-primary, #3b82f6);
  color: white;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.btn {
  padding: 8px 16px;
  border-radius: 6px;
  border: none;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s;
}

.btn-blue {
  background: var(--color-primary, #3b82f6);
  color: white;
}

.btn-blue:hover {
  filter: brightness(1.1);
}

.dark .btn-blue:hover {
  background: #2563eb;
}

.btn-green {
  background: rgba(34, 197, 94, 0.15);
  color: var(--color-success, #22c55e);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.btn-green:hover {
  background: rgba(34, 197, 94, 0.25);
}

.btn-red {
  background: rgba(239, 68, 68, 0.15);
  color: var(--color-danger, #ef4444);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.btn-red:hover {
  background: rgba(239, 68, 68, 0.25);
}

.btn-secondary {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.dark .btn-secondary {
  background: #334155;
}

.btn-secondary:hover {
  background: var(--color-border);
  color: var(--color-text-primary);
}

.dark .btn-secondary:hover {
  background: #475569;
}

.btn-primary {
  background: var(--color-primary, #3b82f6);
  color: white;
}

.btn-primary:hover {
  filter: brightness(1.1);
}

.dark .btn-primary:hover {
  background: #2563eb;
}

.btn-danger {
  background: var(--color-danger, #ef4444);
  color: white;
}

.btn-danger:hover {
  filter: brightness(1.1);
}

.dark .btn-danger:hover {
  background: #dc2626;
}

/* 统计摘要 */
.stats-summary {
  padding: 8px 20px;
  border-bottom: 1px solid var(--color-border);
  font-size: 12px;
  color: var(--color-text-secondary);
  display: flex;
  gap: 16px;
  flex-shrink: 0;
  background: var(--color-bg-secondary);
}

.dark .stats-summary {
  background: rgba(30, 41, 59, 0.2);
}

.stats-summary span {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* 内容区 */
.content-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 0 20px 16px;
  position: relative;
  min-height: 0;
}

/* 批量操作栏 */
.batch-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: rgba(59, 130, 246, 0.08);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 8px;
  margin-bottom: 16px;
}

.batch-bar span {
  font-size: 13px;
  color: var(--color-text-secondary, #94a3b8);
}

.batch-bar span b {
  color: var(--color-primary, #3b82f6);
}

/* 列表视图 */
.sku-table {
  width: 100%;
  border-collapse: collapse;
}

.sku-table th {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border);
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-weight: 500;
  background: var(--color-bg-tertiary);
  position: sticky;
  top: 0;
  z-index: 10;
}

.dark .sku-table th {
  background: rgba(30, 41, 59, 0.95);
}

.sku-table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border-light);
  font-size: 13px;
  color: var(--color-text-primary);
  vertical-align: middle;
}

.dark .sku-table td {
  border-bottom: 1px solid rgba(51, 65, 85, 0.3);
}

.sku-table tr {
  transition: all 0.15s;
  cursor: pointer;
}

.sku-table tr:hover {
  background: var(--color-bg-hover);
}

.dark .sku-table tr:hover {
  background: rgba(255,255,255,0.02);
}

.sku-table tr.selected {
  background: rgba(59, 130, 246, 0.06);
}

.dark .sku-table tr.selected {
  background: rgba(59, 130, 246, 0.06);
}

.sku-table .chk-cell {
  width: 40px;
}

.sku-table .id-cell {
  width: 80px;
  color: var(--color-primary, #3b82f6);
  font-family: monospace;
  font-weight: 600;
}

.sku-table .thumb-cell {
  width: 56px;
}

.sku-table .count-cell {
  width: 80px;
  text-align: center;
}

.sku-table .action-cell {
  width: 100px;
  text-align: right;
}

.sku-thumb {
  width: 40px;
  height: 40px;
  border-radius: 6px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 16px;
  overflow: hidden;
}

.action-btns {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.icon-btn {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary, #94a3b8);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: all 0.2s;
}

.icon-btn:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.icon-btn.edit:hover {
  color: var(--color-primary, #3b82f6);
}

.icon-btn.delete:hover {
  color: var(--color-danger, #ef4444);
}

/* 画廊视图 */
.gallery-view {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}

.gallery-card {
  background: var(--color-bg-primary, #1e293b);
  border: 2px solid transparent;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.gallery-card:hover {
  border-color: var(--color-primary, #3b82f6);
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

.gallery-card.selected {
  border-color: var(--color-primary, #3b82f6);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
}

.gallery-card .chk-wrap {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 5;
}

.gallery-image-box {
  width: 100%;
  height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 40px;
  background: var(--color-bg-tertiary);
  position: relative;
  overflow: hidden;
}

.gallery-image-box .img-count {
  position: absolute;
  bottom: 8px;
  right: 8px;
  background: rgba(0,0,0,0.7);
  color: #fff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
}

.gallery-info {
  padding: 12px;
}

.gallery-sku {
  font-size: 13px;
  color: var(--color-primary, #3b82f6);
  font-weight: 600;
  margin-bottom: 4px;
  font-family: monospace;
}

.gallery-name {
  font-size: 14px;
  color: var(--color-text-primary, #e2e8f0);
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 详情面板 */
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  display: none;
  z-index: 55;
}

.overlay.show {
  display: block;
}

.detail-panel {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: var(--color-bg-primary, #1e293b);
  border-top: 1px solid var(--color-border, #334155);
  border-radius: 16px 16px 0 0;
  box-shadow: 0 -10px 40px rgba(0,0,0,0.5);
  transform: translateY(100%);
  transition: transform 0.3s ease;
  z-index: 60;
  max-height: 60vh;
  display: flex;
  flex-direction: column;
}

.detail-panel.show {
  transform: translateY(0);
}

.detail-panel-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border, #334155);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.detail-panel-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-primary, #e2e8f0);
}

.detail-panel-close {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary, #94a3b8);
  cursor: pointer;
  font-size: 18px;
}

.detail-panel-close:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.detail-panel-body {
  padding: 20px;
  overflow-y: auto;
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 24px;
}

.detail-info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.info-card {
  background: var(--color-bg-tertiary, #0f172a);
  border: 1px solid var(--color-border, #334155);
  border-radius: 8px;
  padding: 12px;
}

.info-card .label {
  font-size: 11px;
  color: var(--color-text-tertiary, #64748b);
  margin-bottom: 4px;
  text-transform: uppercase;
}

.info-card .value {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary, #e2e8f0);
}

.info-full {
  grid-column: 1 / -1;
}

.info-full .value {
  font-size: 13px;
  font-weight: normal;
  color: var(--color-text-secondary, #94a3b8);
}

/* 图片管理 */
.img-manage-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary, #e2e8f0);
  margin-bottom: 12px;
}

.img-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 10px;
}

.img-item {
  aspect-ratio: 1;
  border-radius: 8px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 24px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
}

.img-item:hover .img-del {
  display: flex;
}

.img-del {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(239, 68, 68, 0.9);
  color: #fff;
  border: none;
  font-size: 12px;
  cursor: pointer;
  display: none;
  align-items: center;
  justify-content: center;
}

/* 加载和空状态 */
.loading, .empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--color-text-tertiary, #64748b);
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

/* 模态框 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: var(--color-bg-primary, #1e293b);
  border-radius: 10px;
  width: 480px;
  max-width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-sm {
  width: 400px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border, #334155);
}

.modal-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-primary, #e2e8f0);
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--color-text-tertiary, #64748b);
  padding: 0;
  line-height: 1;
}

.btn-close:hover {
  color: var(--color-text-primary, #e2e8f0);
}

.modal-body {
  padding: 20px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--color-border, #334155);
}

.form-group {
  margin-bottom: 14px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
  color: var(--color-text-primary, #e2e8f0);
  font-weight: 500;
}

.required {
  color: var(--color-danger, #ef4444);
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid var(--color-border, #334155);
  border-radius: 6px;
  font-size: 14px;
  box-sizing: border-box;
  background: var(--color-bg-primary, #1e293b);
  color: var(--color-text-primary, #e2e8f0);
  outline: none;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
  border-color: var(--color-primary, #3b82f6);
}

/* Toast */
.toast {
  position: fixed;
  bottom: 30px;
  right: 30px;
  padding: 12px 24px;
  border-radius: 6px;
  color: white;
  font-size: 14px;
  z-index: 2000;
  animation: slideIn 0.3s ease;
}

.toast.info {
  background: var(--color-primary, #3b82f6);
}

.toast.error {
  background: var(--color-danger, #ef4444);
}

.toast.success {
  background: var(--color-success, #22c55e);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 滚动条 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #475569;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #64748b;
}

/* 分页 */
.pagination-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}
.pagination-info {
  font-size: 12px;
  color: var(--color-text-secondary);
}
.pagination-controls {
  display: flex;
  align-items: center;
  gap: 4px;
}
.page-btn {
  min-width: 28px;
  height: 28px;
  padding: 0 6px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: var(--color-bg-secondary);
  color: var(--color-text-primary);
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}
.page-btn:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}
.page-btn.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: #fff;
}
.page-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.page-ellipsis {
  color: var(--color-text-secondary);
  font-size: 12px;
  padding: 0 2px;
}
</style>
