<template>
  <div class="sku-list-page">
    <PageContainer>
      <div class="main-content" :class="{ expanded: selectedSku }">
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
            <button class="btn btn-secondary" @click="handleExport">📤 导出</button>
            <button class="btn btn-secondary" @click="handleSyncFromCsv">🔄 同步CSV</button>
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
            <p>请通过同步CSV导入SKU数据</p>
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
                  <span :class="['status-badge', sku.status === 'inactive' ? 'inactive' : 'active']">
                    {{ sku.status === 'active' ? '启用' : '禁用' }}
                  </span>
                </td>
                <td>{{ sku.image_count }}</td>
                <td>
                  <button class="btn-icon" @click.stop="openEditDialog(sku)" title="编辑">✏️</button>
                  <button 
                    :class="['btn-icon', { danger: sku.status === 'active' }]" 
                    @click.stop="confirmToggleStatus(sku)" 
                    :title="sku.status === 'active' ? '禁用' : '启用'"
                  >{{ sku.status === 'active' ? '⛔' : '✅' }}</button>
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
                  <span :class="['status-tag', sku.status === 'inactive' ? 'inactive' : 'active']">
                    {{ sku.status === 'active' ? '启用' : '禁用' }}
                  </span>
                  <span class="image-count">📷 {{ sku.image_count }}</span>
                </div>
              </div>
              <div class="gallery-actions">
                <button class="btn-icon" @click.stop="openEditDialog(sku)" title="编辑">✏️</button>
                <button 
                  :class="['btn-icon', { danger: sku.status === 'active' }]" 
                  @click.stop="confirmToggleStatus(sku)" 
                  :title="sku.status === 'active' ? '禁用' : '启用'"
                >{{ sku.status === 'active' ? '⛔' : '✅' }}</button>
              </div>
            </div>
          </div>

          <div class="pagination-bar">
            <div class="selection-info" v-if="selectedSkus.length > 0">
              已选择 {{ selectedSkus.length }} 项
              <button class="btn btn-sm btn-danger" @click="confirmBatchToggleStatus">批量禁用</button>
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
        
        <div class="detail-section">
          <div class="section-title">基本信息</div>
          <div class="detail-grid">
            <div class="detail-item">
              <span class="label">分类</span>
              <span class="value">{{ selectedSku.category || '-' }}</span>
            </div>
            <div class="detail-item">
              <span class="label">状态</span>
              <span :class="['value', selectedSku.status]">
                {{ selectedSku.status === 'active' ? '启用' : '禁用' }}
              </span>
            </div>
            <div class="detail-item">
              <span class="label">图片数量</span>
              <span class="value">{{ selectedSku.image_count }}</span>
            </div>
            <div class="detail-item">
              <span class="label">创建时间</span>
              <span class="value">{{ formatDate(selectedSku.created_at) }}</span>
            </div>
          </div>
          <div class="detail-row-full">
            <span class="label">标签</span>
            <div class="tag-list">
              <span v-for="tag in (selectedSku.tags || '').split(',').filter(t => t.trim())" :key="tag" class="tag">{{ tag.trim() }}</span>
              <span v-if="!selectedSku.tags || !selectedSku.tags.trim()" class="empty-tag">-</span>
            </div>
          </div>
          <div class="detail-row-full">
            <span class="label">描述</span>
            <span class="value-full">{{ selectedSku.description || '-' }}</span>
          </div>
        </div>

        <div class="image-section">
          <div class="section-header">
            <span class="section-title">图片管理</span>
          </div>
          
          <div v-if="currentSkuImages.length > 0" class="image-viewer">
            <div class="image-grid">
              <div 
                v-for="(img, idx) in currentSkuImages" 
                :key="idx"
                class="image-grid-item"
                @click="openImageViewer(img.url, img.filename)"
              >
                <SkuImage :image-path="img" width="100%" height="100%" />
                <div class="image-grid-name">{{ img.filename }}</div>
              </div>
            </div>
            
            <div class="image-stats">
              <span>共 {{ currentSkuImages.length }} 张图片</span>
              <span class="hint">点击图片查看大图</span>
            </div>
          </div>
          
          <div v-else class="empty-images">
            <div class="empty-icon">📷</div>
            <p>暂无图片</p>
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

    <div v-if="showDeleteDialog" class="modal-overlay" @click.self="showDeleteDialog = false">
      <div class="modal modal-sm">
        <div class="modal-header">
          <h3>确认{{ deleteTarget?.status === 'active' || deleteType === 'batch' ? '禁用' : '启用' }}</h3>
          <button class="btn-close" @click="showDeleteDialog = false">×</button>
        </div>
        <div class="modal-body">
          <p v-if="deleteType === 'single'">
            确定要{{ deleteTarget?.status === 'active' ? '禁用' : '启用' }}SKU <strong>{{ deleteTarget?.sku_id }}</strong> 吗？
          </p>
          <p v-else>
            确定要禁用选中的 {{ selectedSkus.length }} 个SKU吗？
          </p>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showDeleteDialog = false">取消</button>
          <button 
            :class="['btn', { 'btn-danger': deleteTarget?.status === 'active' || deleteType === 'batch' }]" 
            @click="handleToggleStatus" 
            :disabled="submitting"
          >
            {{ submitting ? '处理中...' : `确认${deleteTarget?.status === 'active' || deleteType === 'batch' ? '禁用' : '启用'}` }}
          </button>
        </div>
      </div>
    </div>

    </PageContainer>
    
    <div v-if="toast.show" :class="['toast', toast.type]">{{ toast.message }}</div>

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
import PageHeader from '@layout/PageHeader.vue'
import PageContainer from '@layout/PageContainer.vue'
import SkuImage from '@sku/SkuImage.vue'
import ImageViewer from '@ui/ImageViewer.vue'

const viewMode = ref('gallery')
const skus = ref([])
const stats = ref({})
const categories = ref([])
const loading = ref(false)
const submitting = ref(false)


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

const confirmToggleStatus = (item) => {
  deleteType.value = 'single'
  deleteTarget.value = item
  showDeleteDialog.value = true
}

const confirmBatchToggleStatus = () => {
  if (selectedSkus.value.length === 0) return
  deleteType.value = 'batch'
  showDeleteDialog.value = true
}

const handleToggleStatus = async () => {
  submitting.value = true
  try {
    let res
    let targetStatus
    let actionText
    
    if (deleteType.value === 'single') {
      targetStatus = deleteTarget.value.status === 'active' ? 'inactive' : 'active'
      actionText = targetStatus === 'inactive' ? '禁用' : '启用'
      res = await sku.update(deleteTarget.value.sku_id, { status: targetStatus })
    } else {
      targetStatus = 'inactive'
      actionText = '禁用'
      for (const skuId of selectedSkus.value) {
        await sku.update(skuId, { status: targetStatus })
      }
      res = { success: true }
    }
    
    if (res.success || deleteType.value === 'batch') {
      showToast(`${actionText}成功`)
      showDeleteDialog.value = false
      selectedSkus.value = []
      selectAll.value = false
      if (selectedSku.value && (deleteType.value === 'single' || selectedSkus.value.includes(selectedSku.value.sku_id))) {
        if (deleteType.value === 'single') {
          selectedSku.value.status = targetStatus
        } else {
          selectedSku.value = null
          currentSkuImages.value = []
        }
      }
      loadSkus()
      loadStats()
    } else {
      showToast(res.detail || `${actionText}失败`, 'error')
    }
  } catch (err) {
    showToast(`${targetStatus === 'inactive' ? '禁用' : '启用'}失败: ` + (err.detail || err.message), 'error')
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

const downloadTemplate = () => {
  const csv = 'sku_id,sku_name,description,category,tags\n000001,示例商品,这是一个示例描述,电子产品,标签1,标签2'
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'sku_template.csv'
  a.click()
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

const openImageViewer = (imageUrl, imageName = '图片') => {
  viewerImageUrl.value = imageUrl
  viewerImageName.value = imageName
  showImageViewer.value = true
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
  background: var(--color-bg-secondary);
}

.main-content {
  display: flex;
  gap: 20px;
  transition: all 0.3s ease;
}

.main-content.expanded .left-panel {
  flex: 1;
  max-width: calc(33.33% - 10px);
}

.main-content.expanded .right-panel {
  flex: 2;
  width: auto;
  max-width: calc(66.67% - 10px);
}

.main-content:not(.expanded) .left-panel {
  flex: 1;
}

.main-content:not(.expanded) .right-panel {
  width: 450px;
}

.left-panel {
  display: flex;
  flex-direction: column;
  gap: 15px;
  transition: all 0.3s ease;
}

.right-panel {
  background: var(--color-bg-primary);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}

.right-panel-placeholder {
  width: 450px;
  background: var(--color-bg-primary);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  display: flex;
  align-items: center;
  justify-content: center;
}

.panel-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-primary);
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--color-text-tertiary);
  padding: 0;
  line-height: 1;
}

.btn-close:hover { color: var(--color-text-primary); }

.search-bar {
  display: flex;
  gap: 0;
  background: var(--color-bg-primary);
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
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
  background: var(--color-primary);
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
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 14px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
}

.actions-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.view-toggle {
  display: flex;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.view-toggle .btn {
  border-radius: 0;
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
}

.view-toggle .btn.active {
  background: var(--color-primary);
  color: white;
}

.action-buttons {
  display: flex;
  gap: 4px;
}

.btn-icon {
  padding: 8px;
  min-width: 36px;
  min-height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stats-bar {
  display: flex;
  background: var(--color-bg-primary);
  border-radius: var(--radius-md);
  padding: 15px 20px;
  gap: 20px;
  box-shadow: var(--shadow-sm);
}

.stat-item {
  flex: 1;
  text-align: center;
}

.stat-num {
  display: block;
  font-size: 24px;
  font-weight: bold;
  color: var(--color-primary);
}

.stat-item.active .stat-num { color: var(--color-success); }
.stat-item.inactive .stat-num { color: var(--color-text-tertiary); }
.stat-item.images .stat-num { color: var(--color-primary); }

.stat-label {
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.sku-list-container {
  flex: 1;
  background: var(--color-bg-primary);
  border-radius: var(--radius-md);
  padding: 20px;
  box-shadow: var(--shadow-sm);
  overflow-y: auto;
}

.loading, .empty-state {
  text-align: center;
  padding: 40px 20px;
  color: var(--color-text-tertiary);
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
  border-bottom: 1px solid var(--color-border-light);
}

.data-table th {
  background: var(--color-bg-tertiary);
  font-weight: 600;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.data-table td {
  font-size: 13px;
  color: var(--color-text-primary);
  cursor: pointer;
}

.data-table tr:hover {
  background: var(--color-bg-tertiary);
}

.data-table tr.selected {
  background: rgba(102, 126, 234, 0.1);
}

.sku-id {
  font-family: monospace;
  font-weight: 600;
  color: var(--color-primary);
}

.status-badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 10px;
  font-size: 11px;
}

.status-badge.active {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.status-badge.inactive {
  background: var(--color-bg-tertiary);
  color: var(--color-text-tertiary);
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

.btn-icon:hover { background: var(--color-bg-tertiary); }
.btn-icon.danger:hover { background: rgba(245, 108, 108, 0.1); }

.gallery-view {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 15px;
}

.gallery-card {
  background: var(--color-bg-primary);
  border: 2px solid transparent;
  border-radius: var(--radius-lg);
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.gallery-card:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-md);
}

.gallery-card.selected {
  border-color: var(--color-primary);
  background: rgba(102, 126, 234, 0.05);
}

.gallery-image-box {
  height: 120px;
  overflow: hidden;
  background: var(--color-bg-tertiary);
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
  color: var(--color-primary);
  font-size: 13px;
}

.gallery-sku-name {
  font-size: 12px;
  color: var(--color-text-primary);
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
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.status-tag.inactive {
  background: var(--color-bg-tertiary);
  color: var(--color-text-tertiary);
}

.image-count {
  font-size: 11px;
  color: var(--color-text-tertiary);
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

.dark .gallery-actions .btn-icon {
  background: rgba(30, 30, 50, 0.9);
}

.pagination-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--color-border-light);
}

.selection-info {
  color: var(--color-text-secondary);
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
  color: var(--color-text-secondary);
}

.pagination button {
  padding: 5px 12px;
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-primary);
}

.pagination button:hover:not(:disabled) {
  background: var(--color-bg-tertiary);
}

.pagination button:disabled {
  opacity: 0.5;
}

.detail-section {
  padding: 15px 20px;
  border-bottom: 1px solid var(--color-border);
  overflow-y: auto;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border-light);
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.detail-item {
  background: var(--color-bg-tertiary);
  padding: 10px 12px;
  border-radius: var(--radius-sm);
}

.detail-item .label {
  display: block;
  color: var(--color-text-tertiary);
  font-size: 12px;
  margin-bottom: 4px;
}

.detail-item .value {
  display: block;
  color: var(--color-text-primary);
  font-size: 13px;
  font-weight: 500;
}

.detail-item .value.active { color: var(--color-success); }
.detail-item .value.inactive { color: var(--color-text-tertiary); }

.detail-row-full {
  margin-top: 12px;
}

.detail-row-full .label {
  display: block;
  color: var(--color-text-tertiary);
  font-size: 12px;
  margin-bottom: 6px;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag {
  background: rgba(102, 126, 234, 0.1);
  color: var(--color-primary);
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
}

.empty-tag {
  color: var(--color-text-tertiary);
  font-size: 12px;
}

.value-full {
  display: block;
  color: var(--color-text-primary);
  font-size: 13px;
  line-height: 1.5;
  background: var(--color-bg-tertiary);
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  min-height: 44px;
}

.image-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 15px 20px;
  overflow-y: auto;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.section-header .section-title {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.image-viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 10px;
  padding: 10px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-md);
  min-height: 200px;
}

.image-grid-item {
  aspect-ratio: 1;
  border-radius: var(--radius-sm);
  overflow: hidden;
  position: relative;
  cursor: pointer;
  transition: all 0.2s;
  border: 2px solid transparent;
  background: var(--color-bg-secondary);
}

.image-grid-item:hover {
  border-color: var(--color-primary);
  transform: scale(1.02);
}

.image-grid-name {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.7);
  padding: 6px 8px;
  color: white;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.image-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.image-stats .hint {
  color: var(--color-primary);
}

.empty-images {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-md);
  color: var(--color-text-tertiary);
}

.empty-images p {
  margin: 8px 0;
  font-size: 13px;
}

.placeholder-content {
  text-align: center;
  color: var(--color-text-tertiary);
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
  background: var(--color-bg-primary);
  border-radius: 10px;
  width: 480px;
  max-width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-sm { width: 400px; }

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
}

.modal-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-primary);
}

.modal-body {
  padding: 20px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--color-border);
}

.form-group {
  margin-bottom: 14px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
  color: var(--color-text-primary);
  font-weight: 500;
}

.required {
  color: var(--color-danger);
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 14px;
  box-sizing: border-box;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
  outline: none;
  border-color: var(--color-primary);
}

.upload-area {
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-lg);
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}

.upload-area:hover {
  border-color: var(--color-primary);
}

.upload-icon {
  display: block;
  font-size: 48px;
  margin-bottom: 12px;
}

.upload-text {
  display: block;
  font-size: 14px;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.upload-hint {
  display: block;
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.import-instructions {
  background: var(--color-bg-tertiary);
  padding: 15px;
  border-radius: var(--radius-sm);
  margin-bottom: 15px;
  font-size: 13px;
}

.import-instructions ul {
  margin: 10px 0;
  padding-left: 20px;
}

.import-instructions code {
  background: var(--color-bg-secondary);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
}

.download-link {
  color: var(--color-primary);
  text-decoration: underline;
}

.file-input {
  width: 100%;
  padding: 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  background: var(--color-bg-primary);
}

.toast {
  position: fixed;
  bottom: 30px;
  right: 30px;
  padding: 12px 24px;
  border-radius: var(--radius-sm);
  color: white;
  font-size: 14px;
  z-index: 2000;
  animation: slideIn 0.3s ease;
}

.toast.info { background: var(--color-primary); }
.toast.error { background: var(--color-danger); }
.toast.success { background: var(--color-success); }

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

@media (max-width: 1100px) {
  .main-content {
    flex-direction: column;
  }
  
  .main-content.expanded .left-panel,
  .main-content:not(.expanded) .left-panel {
    max-width: 100%;
  }
  
  .main-content.expanded .right-panel {
    max-width: 100%;
  }
  
  .right-panel, .right-panel-placeholder {
    width: 100%;
    max-height: none;
  }
  
  .right-panel {
    max-height: calc(100vh - 200px);
  }
}
</style>
