<template>
  <div class="sku-list-page">
    <div class="header">
      <h1>📦 SKU管理</h1>
    </div>

    <div class="main-content">
      <div class="left-panel">
        <div class="panel-header">
          <div class="search-bar">
            <input
              v-model="searchQuery"
              type="text"
              placeholder="搜索SKU编号或名称..."
              class="search-input"
              @keyup.enter="handleSearch"
            />
            <button class="btn btn-search" @click="handleSearch">🔍</button>
          </div>
          <div class="filter-row">
            <select v-model="categoryFilter" class="filter-select" @change="loadSkus">
              <option value="">全部分类</option>
              <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
            </select>
            <select v-model="statusFilter" class="filter-select" @change="loadSkus">
              <option value="">全部状态</option>
              <option value="active">启用</option>
              <option value="inactive">禁用</option>
            </select>
          </div>
        </div>

        <div class="actions-row">
          <div class="view-toggle">
            <button 
              :class="['btn btn-sm', { active: viewMode === 'list' }]" 
              @click="viewMode = 'list'"
            >📋 列表</button>
            <button 
              :class="['btn btn-sm', { active: viewMode === 'gallery' }]" 
              @click="viewMode = 'gallery'"
            >🖼️ 画廊</button>
          </div>
          <div class="action-buttons">
            <button class="btn btn-secondary" @click="showImportDialog = true">📥 导入</button>
            <button class="btn btn-secondary" @click="handleExport">📤 导出</button>
            <button class="btn btn-secondary" @click="handleSyncFromCsv">🔄 同步CSV</button>
            <button class="btn btn-primary" @click="openCreateDialog">➕ 新增SKU</button>
          </div>
        </div>

        <div class="stats-bar">
          <div class="stat-item">
            <span class="stat-num">{{ stats.total_skus || 0 }}</span>
            <span class="stat-label">SKU总数</span>
          </div>
          <div class="stat-item active">
            <span class="stat-num">{{ stats.active_skus || 0 }}</span>
            <span class="stat-label">启用</span>
          </div>
          <div class="stat-item inactive">
            <span class="stat-num">{{ stats.inactive_skus || 0 }}</span>
            <span class="stat-label">禁用</span>
          </div>
          <div class="stat-item images">
            <span class="stat-num">{{ stats.total_images || 0 }}</span>
            <span class="stat-label">图片数</span>
          </div>
        </div>

        <div class="sku-list-container">
          <div v-if="loading" class="loading">加载中...</div>

          <div v-else-if="skus.length === 0" class="empty-state">
            <div class="empty-icon">📭</div>
            <p>暂无SKU数据</p>
            <button class="btn btn-primary" @click="openCreateDialog">新增第一个SKU</button>
          </div>

          <table v-else-if="viewMode === 'list'" class="data-table">
            <thead>
              <tr>
                <th><input type="checkbox" v-model="selectAll" @change="toggleSelectAll" /></th>
                <th>SKU编号</th>
                <th>名称</th>
                <th>分类</th>
                <th>状态</th>
                <th>图片数</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr 
                v-for="sku in skus" 
                :key="sku.id" 
                :class="{ selected: selectedSku?.sku_id === sku.sku_id }"
                @click="selectSku(sku)"
              >
                <td><input type="checkbox" :value="sku.sku_id" v-model="selectedSkus" @click.stop /></td>
                <td class="sku-id">{{ sku.sku_id }}</td>
                <td>{{ sku.sku_name }}</td>
                <td>{{ sku.category || '-' }}</td>
                <td>
                  <span :class="['status-badge', sku.status]">
                    {{ sku.status === 'active' ? '启用' : '禁用' }}
                  </span>
                </td>
                <td>{{ sku.image_count }}</td>
                <td>
                  <button class="btn-icon" @click.stop="openEditDialog(sku)" title="编辑">✏️</button>
                  <button class="btn-icon danger" @click.stop="confirmDelete(sku)" title="删除">🗑️</button>
                </td>
              </tr>
            </tbody>
          </table>

          <div v-else class="gallery-view">
            <div 
              v-for="sku in skus" 
              :key="sku.id" 
              :class="['gallery-card', { selected: selectedSku?.sku_id === sku.sku_id }]"
              @click="selectSku(sku)"
            >
              <div class="gallery-image-box">
                <SkuImage 
                  :image-path="getFirstImagePath(sku.sku_id)" 
                  width="100%"
                  height="100%"
                />
                <div v-if="sku.image_count > 1" class="img-count">{{ sku.image_count }}</div>
              </div>
              <div class="gallery-info">
                <div class="gallery-sku-id">{{ sku.sku_id }}</div>
                <div class="gallery-sku-name">{{ sku.sku_name }}</div>
                <div class="gallery-status">
                  <span :class="['status-tag', sku.status]">
                    {{ sku.status === 'active' ? '启用' : '禁用' }}
                  </span>
                  <span class="image-count">📷 {{ sku.image_count }}</span>
                </div>
              </div>
              <div class="gallery-actions">
                <button class="btn-icon" @click.stop="openEditDialog(sku)" title="编辑">✏️</button>
                <button class="btn-icon danger" @click.stop="confirmDelete(sku)" title="删除">🗑️</button>
              </div>
            </div>
          </div>

          <div class="pagination-bar">
            <div class="selection-info" v-if="selectedSkus.length > 0">
              已选择 {{ selectedSkus.length }} 项
              <button class="btn btn-sm btn-danger" @click="confirmBatchDelete">批量删除</button>
            </div>
            <div class="pagination">
              <button :disabled="page <= 1" @click="changePage(page - 1)">上一页</button>
              <span>第 {{ page }} / {{ totalPages }} 页</span>
              <button :disabled="page >= totalPages" @click="changePage(page + 1)">下一页</button>
            </div>
          </div>
        </div>
      </div>

      <div class="right-panel" v-if="selectedSku">
        <div class="panel-header">
          <h3>{{ selectedSku.sku_id }} - {{ selectedSku.sku_name }}</h3>
          <button class="btn-close" @click="selectedSku = null">×</button>
        </div>
        
        <div class="sku-detail">
          <div class="detail-row">
            <span class="label">分类</span>
            <span class="value">{{ selectedSku.category || '-' }}</span>
          </div>
          <div class="detail-row">
            <span class="label">状态</span>
            <span :class="['value', selectedSku.status]">
              {{ selectedSku.status === 'active' ? '启用' : '禁用' }}
            </span>
          </div>
          <div class="detail-row">
            <span class="label">图片数量</span>
            <span class="value">{{ selectedSku.image_count }}</span>
          </div>
          <div class="detail-row">
            <span class="label">标签</span>
            <span class="value">{{ selectedSku.tags || '-' }}</span>
          </div>
          <div class="detail-row">
            <span class="label">描述</span>
            <span class="value">{{ selectedSku.description || '-' }}</span>
          </div>
          <div class="detail-row">
            <span class="label">创建时间</span>
            <span class="value">{{ formatDate(selectedSku.created_at) }}</span>
          </div>
        </div>

        <div class="image-viewer">
          <div class="viewer-header">
            <span>图片浏览</span>
            <button class="btn btn-sm btn-primary" @click="openUploadDialog">📤 上传图片</button>
          </div>
          
          <div class="image-slider" v-if="currentSkuImages.length > 0">
            <div class="slider-container">
              <SkuImage 
                :image-path="currentSkuImages[currentImageIndex]" 
                width="100%"
                height="100%"
              />
              <div class="slide-info">
                <span>{{ currentSkuImages[currentImageIndex]?.filename }}</span>
                <button class="btn btn-xs btn-danger" @click="deleteImage(currentImageIndex)">删除</button>
              </div>
            </div>
            
            <div class="slider-controls">
              <button class="btn btn-nav" @click="prevImage" :disabled="currentImageIndex <= 0">‹</button>
              <div class="slider-dots">
                <span 
                  v-for="(_, idx) in currentSkuImages" 
                  :key="idx"
                  :class="['dot', { active: idx === currentImageIndex }]"
                  @click="goToImage(idx)"
                ></span>
              </div>
              <button class="btn btn-nav" @click="nextImage" :disabled="currentImageIndex >= currentSkuImages.length - 1">›</button>
            </div>
            
            <div class="image-counter">{{ currentImageIndex + 1 }} / {{ currentSkuImages.length }}</div>
          </div>
          
          <div v-else class="empty-images">
            <div class="empty-icon">📷</div>
            <p>暂无图片</p>
            <button class="btn btn-primary" @click="openUploadDialog">上传图片</button>
          </div>
        </div>
      </div>

      <div class="right-panel-placeholder" v-else>
        <div class="placeholder-content">
          <span class="placeholder-icon">👆</span>
          <p>选择一个SKU查看详情和图片</p>
        </div>
      </div>
    </div>

    <div v-if="showCreateDialog || showEditDialog" class="modal-overlay" @click.self="closeDialog">
      <div class="modal">
        <div class="modal-header">
          <h3>{{ showEditDialog ? '编辑SKU' : '新增SKU' }}</h3>
          <button class="btn-close" @click="closeDialog">×</button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>SKU编号 <span class="required">*</span></label>
            <input
              v-model="formData.sku_id"
              type="text"
              :disabled="showEditDialog"
              placeholder="请输入SKU编号"
              maxlength="50"
            />
          </div>
          <div class="form-group">
            <label>SKU名称 <span class="required">*</span></label>
            <input
              v-model="formData.sku_name"
              type="text"
              placeholder="请输入SKU名称"
              maxlength="200"
            />
          </div>
          <div class="form-group">
            <label>分类</label>
            <input v-model="formData.category" type="text" placeholder="请输入分类" />
          </div>
          <div class="form-group">
            <label>描述</label>
            <textarea v-model="formData.description" placeholder="请输入描述" rows="3"></textarea>
          </div>
          <div class="form-group">
            <label>标签</label>
            <input v-model="formData.tags" type="text" placeholder="多个标签用逗号分隔" />
          </div>
          <div class="form-group" v-if="showEditDialog">
            <label>状态</label>
            <select v-model="formData.status">
              <option value="active">启用</option>
              <option value="inactive">禁用</option>
            </select>
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

    <div v-if="showUploadDialog" class="modal-overlay" @click.self="showUploadDialog = false">
      <div class="modal">
        <div class="modal-header">
          <h3>上传图片 - {{ selectedSku?.sku_id }}</h3>
          <button class="btn-close" @click="showUploadDialog = false">×</button>
        </div>
        <div class="modal-body">
          <div class="upload-area" @click="triggerFileUpload">
            <input type="file" accept="image/*" multiple ref="imageUpload" @change="handleImageUpload" hidden />
            <span class="upload-icon">📤</span>
            <span class="upload-text">点击或拖拽上传图片</span>
            <span class="upload-hint">支持 JPG、PNG、BMP 格式</span>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showUploadDialog = false">取消</button>
        </div>
      </div>
    </div>

    <div v-if="showDeleteDialog" class="modal-overlay" @click.self="showDeleteDialog = false">
      <div class="modal modal-sm">
        <div class="modal-header">
          <h3>确认删除</h3>
          <button class="btn-close" @click="showDeleteDialog = false">×</button>
        </div>
        <div class="modal-body">
          <p v-if="deleteType === 'single'">
            确定要删除SKU <strong>{{ deleteTarget?.sku_id }}</strong> 吗？此操作不可恢复。
          </p>
          <p v-else>
            确定要删除选中的 {{ selectedSkus.length }} 个SKU吗？此操作不可恢复。
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

    <div v-if="showImportDialog" class="modal-overlay" @click.self="showImportDialog = false">
      <div class="modal">
        <div class="modal-header">
          <h3>导入CSV</h3>
          <button class="btn-close" @click="showImportDialog = false">×</button>
        </div>
        <div class="modal-body">
          <div class="import-instructions">
            <p>请上传CSV文件，文件格式要求：</p>
            <ul>
              <li>必须包含 <code>sku_id</code> 和 <code>sku_name</code> 列</li>
              <li>可选列：<code>description</code>, <code>category</code>, <code>tags</code></li>
            </ul>
            <a href="#" @click.prevent="downloadTemplate" class="download-link">下载模板文件</a>
          </div>
          <input type="file" accept=".csv" @change="handleFileChange" ref="fileInput" class="file-input" />
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showImportDialog = false">取消</button>
          <button class="btn btn-primary" @click="handleImport" :disabled="!importFile || importing">
            {{ importing ? '导入中...' : '开始导入' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="toast.show" :class="['toast', toast.type]">{{ toast.message }}</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { sku } from '../api/client'
import SkuImage from './result/SkuImage.vue'

const viewMode = ref('gallery')
const skus = ref([])
const stats = ref({})
const categories = ref([])
const loading = ref(false)
const submitting = ref(false)
const importing = ref(false)

const searchQuery = ref('')
const categoryFilter = ref('')
const statusFilter = ref('')
const page = ref(1)
const pageSize = ref(20)
const total = ref(0)

const selectedSkus = ref([])
const selectAll = ref(false)
const selectedSku = ref(null)
const currentSkuImages = ref([])
const currentImageIndex = ref(0)

const showCreateDialog = ref(false)
const showEditDialog = ref(false)
const showDeleteDialog = ref(false)
const showImportDialog = ref(false)
const showUploadDialog = ref(false)

const deleteType = ref('single')
const deleteTarget = ref(null)
const importFile = ref(null)

const imageUpload = ref(null)

const formData = ref({
  sku_id: '',
  sku_name: '',
  description: '',
  category: '',
  tags: '',
  status: 'active'
})

const toast = ref({ show: false, message: '', type: 'info' })

const skuImagesCache = ref({})

const totalPages = computed(() => Math.ceil(total.value / pageSize.value) || 1)

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
    const res = await sku.list(page.value, pageSize.value, searchQuery.value, categoryFilter.value, statusFilter.value)
    if (res.success) {
      skus.value = res.skus
      total.value = res.total
      await preloadSkuImages()
      await loadStats()
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
        console.error(`Failed to load images for ${s.sku_id}:`, err)
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

const loadCategories = async () => {
  try {
    const res = await sku.getCategories()
    if (res.success) {
      categories.value = res.categories
    }
  } catch (err) {
    console.error('Failed to load categories:', err)
  }
}

const handleSearch = () => {
  page.value = 1
  loadSkus()
}

const changePage = (newPage) => {
  if (newPage >= 1 && newPage <= totalPages.value) {
    page.value = newPage
    loadSkus()
  }
}

const selectSku = async (skuItem) => {
  selectedSku.value = skuItem
  currentImageIndex.value = 0
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

const openCreateDialog = () => {
  formData.value = {
    sku_id: '',
    sku_name: '',
    description: '',
    category: '',
    tags: '',
    status: 'active'
  }
  showCreateDialog.value = true
}

const openEditDialog = (item) => {
  formData.value = {
    sku_id: item.sku_id,
    sku_name: item.sku_name,
    description: item.description || '',
    category: item.category || '',
    tags: item.tags || '',
    status: item.status
  }
  showEditDialog.value = true
}

const openUploadDialog = () => {
  showUploadDialog.value = true
}

const closeDialog = () => {
  showCreateDialog.value = false
  showEditDialog.value = false
  showImportDialog.value = false
  showUploadDialog.value = false
  importFile.value = null
}

const handleSubmit = async () => {
  if (!formData.value.sku_id.trim()) {
    showToast('SKU编号不能为空', 'error')
    return
  }
  if (!formData.value.sku_name.trim()) {
    showToast('SKU名称不能为空', 'error')
    return
  }

  submitting.value = true
  try {
    let res
    if (showEditDialog.value) {
      res = await sku.update(formData.value.sku_id, formData.value)
    } else {
      res = await sku.create(formData.value)
    }
    if (res.sku_id || res.success) {
      showToast(showEditDialog.value ? '更新成功' : '创建成功')
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

const confirmBatchDelete = () => {
  if (selectedSkus.value.length === 0) return
  deleteType.value = 'batch'
  showDeleteDialog.value = true
}

const handleDelete = async () => {
  submitting.value = true
  try {
    let res
    if (deleteType.value === 'single') {
      res = await sku.delete(deleteTarget.value.sku_id)
    } else {
      res = await sku.batchDelete(selectedSkus.value)
    }
    if (res.success) {
      showToast(res.message || '删除成功')
      showDeleteDialog.value = false
      selectedSkus.value = []
      selectAll.value = false
      if (selectedSku.value && (deleteType.value === 'single' || selectedSkus.value.includes(selectedSku.value.sku_id))) {
        selectedSku.value = null
        currentSkuImages.value = []
      }
      loadSkus()
      loadStats()
    } else {
      showToast(res.detail || '删除失败', 'error')
    }
  } catch (err) {
    showToast('删除失败: ' + (err.detail || err.message), 'error')
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

const handleExport = async () => {
  try {
    await sku.exportCsv()
    showToast('导出成功')
  } catch (err) {
    showToast('导出失败', 'error')
  }
}

const handleSyncFromCsv = async () => {
  try {
    const res = await sku.syncFromCsv()
    if (res.success) {
      showToast(res.message)
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

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    importFile.value = file
  }
}

const handleImport = async () => {
  if (!importFile.value) return

  importing.value = true
  try {
    const res = await sku.importCsv(importFile.value)
    if (res.success) {
      showToast(res.message)
      showImportDialog.value = false
      importFile.value = null
      loadSkus()
      loadStats()
      loadCategories()
    } else {
      showToast(res.detail || '导入失败', 'error')
    }
  } catch (err) {
    showToast('导入失败: ' + err.message, 'error')
  } finally {
    importing.value = false
  }
}

const downloadTemplate = () => {
  const csv = 'sku_id,sku_name,description,category,tags\n000001,示例商品,这是一个示例描述,电子产品,标签1,标签2'
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'sku_template.csv'
  a.click()
}

const triggerFileUpload = () => {
  imageUpload.value?.click()
}

const handleImageUpload = async (event) => {
  const files = Array.from(event.target.files)
  if (files.length === 0) return

  try {
    const res = await sku.uploadImages(selectedSku.value.sku_id, files)
    if (res.success) {
      showToast(`成功上传 ${files.length} 张图片`)
      showUploadDialog.value = false
      await loadSkuImages(selectedSku.value.sku_id)
      loadSkus()
      loadStats()
    } else {
      showToast(res.detail || '上传失败', 'error')
    }
  } catch (err) {
    showToast('上传失败: ' + err.message, 'error')
  }
  
  event.target.value = ''
}

const deleteImage = async (index) => {
  const image = currentSkuImages.value[index]
  if (!image) return

  try {
    const res = await sku.deleteImage(selectedSku.value.sku_id, image.filename)
    if (res.success) {
      showToast('删除成功')
      await loadSkuImages(selectedSku.value.sku_id)
      if (currentImageIndex.value >= currentSkuImages.value.length) {
        currentImageIndex.value = Math.max(0, currentSkuImages.value.length - 1)
      }
      loadSkus()
      loadStats()
    } else {
      showToast(res.detail || '删除失败', 'error')
    }
  } catch (err) {
    showToast('删除失败: ' + err.message, 'error')
  }
}

const prevImage = () => {
  if (currentImageIndex.value > 0) {
    currentImageIndex.value--
  }
}

const nextImage = () => {
  if (currentImageIndex.value < currentSkuImages.value.length - 1) {
    currentImageIndex.value++
  }
}

const goToImage = (index) => {
  currentImageIndex.value = index
}

watch(selectedSku, () => {
  currentImageIndex.value = 0
})

onMounted(() => {
  loadSkus()
  loadStats()
  loadCategories()
})
</script>

<style scoped>
.sku-list-page {
  min-height: 100vh;
  background: #f5f7fa;
}

.header {
  background: linear-gradient(135deg, #4a69bd 0%, #6a89cc 100%);
  padding: 20px 30px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header h1 {
  margin: 0;
  font-size: 22px;
  color: white;
  font-weight: 600;
}

.main-content {
  display: flex;
  gap: 20px;
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.left-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.right-panel {
  width: 400px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.right-panel-placeholder {
  width: 400px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
}

.panel-header {
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h3 {
  margin: 0;
  font-size: 16px;
  color: #333;
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
  padding: 0;
  line-height: 1;
}

.btn-close:hover { color: #333; }

.search-bar {
  display: flex;
  gap: 0;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.search-input {
  flex: 1;
  padding: 12px 16px;
  border: none;
  font-size: 14px;
  outline: none;
}

.btn-search {
  padding: 12px 16px;
  border: none;
  background: #4a69bd;
  color: white;
  cursor: pointer;
  font-size: 16px;
}

.filter-row {
  display: flex;
  gap: 10px;
}

.filter-select {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  font-size: 14px;
  background: white;
}

.actions-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.view-toggle {
  display: flex;
  background: #f0f0f0;
  border-radius: 6px;
  overflow: hidden;
}

.view-toggle .btn {
  border-radius: 0;
  border: none;
  background: transparent;
}

.view-toggle .btn.active {
  background: #4a69bd;
  color: white;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 8px 14px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.btn-xs {
  padding: 4px 8px;
  font-size: 11px;
}

.btn-primary {
  background: #4a69bd;
  color: white;
}

.btn-primary:hover { background: #3d5a9e; }
.btn-primary:disabled { background: #a5a5a5; cursor: not-allowed; }

.btn-secondary {
  background: #e8e8e8;
  color: #333;
}

.btn-secondary:hover { background: #d8d8d8; }

.btn-danger {
  background: #e74c3c;
  color: white;
}

.btn-danger:hover { background: #c0392b; }

.btn-nav {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #f0f0f0;
  border: none;
  font-size: 20px;
  color: #666;
}

.btn-nav:hover:not(:disabled) { background: #e0e0e0; }
.btn-nav:disabled { opacity: 0.4; }

.stats-bar {
  display: flex;
  background: white;
  border-radius: 8px;
  padding: 15px 20px;
  gap: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stat-item {
  flex: 1;
  text-align: center;
}

.stat-num {
  display: block;
  font-size: 24px;
  font-weight: bold;
  color: #4a69bd;
}

.stat-item.active .stat-num { color: #27ae60; }
.stat-item.inactive .stat-num { color: #95a5a6; }
.stat-item.images .stat-num { color: #3498db; }

.stat-label {
  font-size: 12px;
  color: #888;
}

.sku-list-container {
  flex: 1;
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  overflow-y: auto;
}

.loading, .empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: 10px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

.data-table th {
  background: #f8f9fa;
  font-weight: 600;
  color: #666;
  font-size: 13px;
}

.data-table td {
  font-size: 13px;
  color: #555;
  cursor: pointer;
}

.data-table tr:hover {
  background: #f8f9fa;
}

.data-table tr.selected {
  background: #e8f4fd;
}

.sku-id {
  font-family: monospace;
  font-weight: 600;
  color: #4a69bd;
}

.status-badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 10px;
  font-size: 11px;
}

.status-badge.active {
  background: #e8f5e9;
  color: #27ae60;
}

.status-badge.inactive {
  background: #f5f5f5;
  color: #95a5a6;
}

.btn-icon {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 4px;
  border-radius: 4px;
  transition: background 0.2s;
}

.btn-icon:hover { background: #f0f0f0; }
.btn-icon.danger:hover { background: #ffe6e6; }

.gallery-view {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 15px;
}

.gallery-card {
  background: white;
  border: 2px solid transparent;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.gallery-card:hover {
  border-color: #4a69bd;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.gallery-card.selected {
  border-color: #4a69bd;
  background: #f8fafc;
}

.gallery-image-box {
  height: 120px;
  overflow: hidden;
  background: #f8f9fa;
  position: relative;
}

.img-count {
  position: absolute;
  bottom: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
}

.gallery-info {
  padding: 10px;
}

.gallery-sku-id {
  font-family: monospace;
  font-weight: 600;
  color: #4a69bd;
  font-size: 13px;
}

.gallery-sku-name {
  font-size: 12px;
  color: #333;
  margin: 4px 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gallery-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-tag {
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 10px;
}

.status-tag.active {
  background: #e8f5e9;
  color: #27ae60;
}

.status-tag.inactive {
  background: #f5f5f5;
  color: #95a5a6;
}

.image-count {
  font-size: 11px;
  color: #999;
}

.gallery-actions {
  position: absolute;
  top: 6px;
  right: 6px;
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.gallery-card:hover .gallery-actions {
  opacity: 1;
}

.gallery-actions .btn-icon {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(4px);
}

.pagination-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #f0f0f0;
}

.selection-info {
  color: #666;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.pagination {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: #666;
}

.pagination button {
  padding: 5px 12px;
  border: 1px solid #e0e0e0;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.pagination button:hover:not(:disabled) {
  background: #f5f5f5;
}

.pagination button:disabled {
  opacity: 0.5;
}

.sku-detail {
  padding: 15px 20px;
  border-bottom: 1px solid #eee;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px dashed #f0f0f0;
}

.detail-row:last-child {
  border-bottom: none;
}

.detail-row .label {
  color: #888;
  font-size: 13px;
}

.detail-row .value {
  color: #333;
  font-size: 13px;
}

.detail-row .value.active { color: #27ae60; }
.detail-row .value.inactive { color: #95a5a6; }

.image-viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 15px;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.image-slider {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.slider-container {
  flex: 1;
  background: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  min-height: 300px;
}

.slide-info {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.6);
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: white;
  font-size: 12px;
}

.slider-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 15px;
  margin-top: 15px;
}

.slider-dots {
  display: flex;
  gap: 8px;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ddd;
  cursor: pointer;
  transition: all 0.2s;
}

.dot.active {
  background: #4a69bd;
  transform: scale(1.2);
}

.image-counter {
  text-align: center;
  font-size: 12px;
  color: #888;
  margin-top: 10px;
}

.empty-images {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #f8f9fa;
  border-radius: 8px;
  color: #999;
}

.empty-images p {
  margin: 8px 0;
  font-size: 13px;
}

.placeholder-content {
  text-align: center;
  color: #bbb;
  padding: 60px 20px;
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

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
  background: white;
  border-radius: 10px;
  width: 480px;
  max-width: 90%;
  max-height: 90vh;
  overflow-y: auto;
}

.modal-sm { width: 400px; }

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
}

.modal-header h3 {
  margin: 0;
  font-size: 16px;
  color: #333;
}

.modal-body {
  padding: 20px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid #eee;
}

.form-group {
  margin-bottom: 14px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
  color: #333;
  font-weight: 500;
}

.required {
  color: #e74c3c;
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  font-size: 14px;
  box-sizing: border-box;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
  outline: none;
  border-color: #4a69bd;
}

.upload-area {
  border: 2px dashed #ddd;
  border-radius: 10px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}

.upload-area:hover {
  border-color: #4a69bd;
}

.upload-icon {
  display: block;
  font-size: 48px;
  margin-bottom: 12px;
}

.upload-text {
  display: block;
  font-size: 14px;
  color: #333;
  margin-bottom: 4px;
}

.upload-hint {
  display: block;
  font-size: 12px;
  color: #999;
}

.import-instructions {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 15px;
  font-size: 13px;
}

.import-instructions ul {
  margin: 10px 0;
  padding-left: 20px;
}

.import-instructions code {
  background: #e8e8e8;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}

.download-link {
  color: #4a69bd;
  text-decoration: underline;
}

.file-input {
  width: 100%;
  padding: 10px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
}

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

.toast.info { background: #4a69bd; }
.toast.error { background: #e74c3c; }
.toast.success { background: #27ae60; }

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

@media (max-width: 1000px) {
  .main-content {
    flex-direction: column;
  }
  
  .right-panel, .right-panel-placeholder {
    width: 100%;
  }
}
</style>
