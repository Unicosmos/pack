<template>
  <div v-if="task" class="review-container" :class="{ 'inline': inline }">
    <div class="review-header">
      <h4>识别结果审核</h4>
      <span class="header-hint">已识别 {{ boxes.length }} 个箱体 · 点击输入框可修改匹配</span>
    </div>
    
    <div 
      class="review-boxes-section"
      ref="scrollContainer"
      @wheel="handleWheel"
    >
      <div class="review-boxes-list">
        <div
          v-for="(box, idx) in boxes"
          :key="box.box_id"
          class="review-box-item"
          :class="{ 
            'rejected': box.status === 'rejected', 
            'deleted': box.status === 'deleted',
            'has-match': box.match_result && box.match_result.sku_id
          }"
        >
          <div class="review-box-header">
            <span class="box-index">箱体 {{ idx + 1 }}</span>
            <span class="box-conf">置信度: {{ (box.confidence * 100).toFixed(1) }}%</span>
          </div>
          
          <div class="review-crop-wrapper" @click="openLargeImage(getCropImageUrl(box))">
            <SkuImage
              :image-path="getCropImageUrl(box)"
              height="120px"
              class="review-crop"
            />
            <div class="click-hint-small">👆 查看大图</div>
          </div>

          <div v-if="box.match_result" class="review-box-match">
            <div class="match-info">
              <span class="match-label">匹配结果:</span>
              <div class="match-main">
                <span class="match-value" :class="box.match_result.status">
                  {{ box.match_result.sku_id || '未匹配' }}
                </span>
                <span v-if="box.match_result.sku_name" class="match-name">
                  {{ box.match_result.sku_name }}
                </span>
              </div>
              <span v-if="box.match_result.similarity" class="match-conf">
                相似度: {{ (box.match_result.similarity * 100).toFixed(1) }}%
              </span>
            </div>
            
            <input
              type="text"
              class="sku-input"
              :class="{ modified: box.isModified }"
              v-model="box.custom_sku"
              @input="handleCustomSkuInput(box)"
              placeholder="输入自定义SKU"
              title="修改匹配SKU"
            />

            <div v-if="box.match_result.top5_labels && box.match_result.top5_labels.length > 0" class="top5-section">
              <div class="top5-header" @click="toggleTop5(box)">
                <div class="top5-title">
                  <span class="top5-expand-icon">{{ box.showTop5 ? '▼' : '▶' }}</span>
                  <span>Top 5 匹配结果</span>
                </div>
                <span class="top5-hint">点击展开/收起</span>
              </div>
              <div v-show="box.showTop5" class="top5-list-container">
                <div class="top5-list">
                  <div
                    v-for="(label, lIdx) in box.match_result.top5_labels.slice(0, 5)"
                    :key="lIdx"
                    class="top5-item"
                    :class="{ 'selected': box.custom_sku === label.sku_id, 'top1': lIdx === 0 }"
                    @click="selectTop5Sku(box, label)"
                  >
                    <div class="top5-rank" v-if="lIdx === 0">🥇</div>
                    <SkuImage
                      :image-path="label.image_path ? getImageUrlFromPath(label.image_path) : ''"
                      height="40px"
                      class="top5-image"
                      :lazy="true"
                    />
                    <div class="top5-info">
                      <span class="top5-sku">{{ label.sku_id || label.label }}</span>
                      <span class="top5-name">{{ label.name || label.sku_name || '' }}</span>
                    </div>
                    <span class="top5-sim">{{ (label.similarity * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="review-box-footer">
            <div class="modified-indicator" v-if="box.isModified">
              <span class="modified-badge">已修改</span>
            </div>
            <button
              class="btn-delete"
              @click="deleteBox(box)"
              title="删除此检测框"
            >
              🗑️ 删除
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="review-footer">
      <button class="btn btn-primary" @click="handleUpdate">更新</button>
    </div>

    <ImageViewer
      :visible="showImageViewer"
      :image-url="currentImageUrl"
      :image-name="currentImageName"
      @update:visible="showImageViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import SkuImage from '@sku/SkuImage.vue'
import ImageViewer from '@ui/ImageViewer.vue'
import { getImageUrlFromPath } from '@api/client'

const props = defineProps({
  task: { type: Object, default: null },
  inline: { type: Boolean, default: false }
})

const emit = defineEmits(['cancel', 'update'])

const boxes = ref([])
const scrollContainer = ref(null)

const showImageViewer = ref(false)
const currentImageUrl = ref('')
const currentImageName = ref('')

const getTaskImageUrl = () => {
  if (!props.task) return ''
  return `/api/tasks/${props.task.id}/image`
}

const getCropImageUrl = (box) => {
  if (!box) return ''
  if (box.crop_base64) {
    return { url: 'data:image/jpeg;base64,' + box.crop_base64 }
  }
  if (box.crop_path) {
    return getImageUrlFromPath(box.crop_path)
  }
  return ''
}

const initData = () => {
  if (!props.task) return
  
  boxes.value = []
  
  let sourceBoxes = []
  
  if (props.task.result?.detections?.boxes) {
    sourceBoxes = props.task.result.detections.boxes
  } else if (props.task.boxes) {
    sourceBoxes = props.task.boxes
  } else if (props.task.result?.boxes) {
    sourceBoxes = props.task.result.boxes
  }
  
  boxes.value = sourceBoxes.map((box, idx) => ({
    ...box,
    box_id: box.box_id || `box_${idx}`,
    status: box.status || 'approved',
    custom_sku: box.custom_sku || '',
    isModified: false,
    isDeleted: false,
    _originalMatchResult: box.match_result ? { ...box.match_result } : null
  }))
}

onMounted(() => {
  if (props.task) {
    initData()
  }
})

watch(() => props.task?.id, (newId, oldId) => {
  if (newId && newId !== oldId) {
    initData()
  }
})

const toggleTop5 = (box) => {
  box.showTop5 = !box.showTop5
}

const selectTop5Sku = (box, label) => {
  const oldSku = box.match_result?.sku_id
  const newSku = label.sku_id || label.label
  
  box.custom_sku = newSku
  if (box.match_result) {
    box.match_result.sku_id = newSku
    box.match_result.sku_name = label.name || label.sku_name
    box.match_result.status = 'matched'
  }
  
  if (oldSku !== newSku) {
    box.isModified = true
  }
}

const handleCustomSkuInput = (box) => {
  const sku = box.custom_sku?.trim()
  
  if (!sku) {
    box.isModified = false
    if (box.match_result) {
      const originalMatch = box._originalMatchResult
      if (originalMatch) {
        box.match_result.sku_id = originalMatch.sku_id
        box.match_result.sku_name = originalMatch.sku_name
        box.match_result.status = originalMatch.status
      }
    }
    return
  }
  
  if (box.match_result) {
    const originalSku = box._originalMatchResult?.sku_id
    if (sku !== originalSku) {
      box.isModified = true
      box.match_result.sku_id = sku
      box.match_result.status = 'matched'
    }
  }
}

const deleteBox = (box) => {
  box.status = 'deleted'
  box.isDeleted = true
}

const handleCancel = () => {
  emit('cancel')
}

const handleUpdate = () => {
  const nonDeletedBoxes = boxes.value.filter(b => !b.isDeleted)
  const modified = nonDeletedBoxes.filter(b => b.isModified).length
  const deleted = boxes.value.length - nonDeletedBoxes.length
  const approved = nonDeletedBoxes.length

  const resultBoxes = nonDeletedBoxes.map((box, idx) => ({
    ...box,
    box_id: `box_${idx}`,
    is_manual_override: box.isModified
  }))

  emit('update', {
    task: props.task,
    boxes: resultBoxes,
    approvedCount: approved,
    rejectedCount: 0,
    deletedCount: deleted
  })

  let msg = `更新成功`
  if (modified > 0) msg += `，修改 ${modified} 个`
  if (deleted > 0) msg += `，删除 ${deleted} 个`
  ElMessage.success(msg)
}

const openLargeImage = (imagePath) => {
  if (!imagePath) return
  
  if (typeof imagePath === 'object' && imagePath.url) {
    currentImageUrl.value = imagePath.url
  } else if (typeof imagePath === 'string') {
    if (imagePath.startsWith('/') || imagePath.startsWith('http') || imagePath.startsWith('data:image')) {
      currentImageUrl.value = imagePath
    } else {
      currentImageUrl.value = getTaskImageUrl()
    }
  }
  
  currentImageName.value = props.task?.image_name || '检测图片'
  showImageViewer.value = true
}

let isAnimating = false

const handleWheel = (e) => {
  if (!scrollContainer.value || isAnimating) return
  
  const container = scrollContainer.value
  const scrollTop = container.scrollTop
  const scrollHeight = container.scrollHeight
  const clientHeight = container.clientHeight
  const deltaY = e.deltaY

  if (scrollTop === 0 && deltaY < 0) {
    e.preventDefault()
    animateScroll(container, -50, 0, 150)
  } else if (scrollTop >= scrollHeight - clientHeight - 1 && deltaY > 0) {
    e.preventDefault()
    animateScroll(container, 50, scrollHeight - clientHeight, 150)
  }
}

const animateScroll = (container, offset, limit, duration) => {
  isAnimating = true
  const start = container.scrollTop
  const startTime = performance.now()
  
  const animate = (currentTime) => {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    
    const easeOut = 1 - Math.pow(1 - progress, 3)
    let newScrollTop = start + offset * easeOut
    
    newScrollTop = Math.max(0, Math.min(newScrollTop, limit))
    container.scrollTop = newScrollTop
    
    if (progress < 1) {
      requestAnimationFrame(animate)
    } else {
      isAnimating = false
    }
  }
  
  requestAnimationFrame(animate)
}

let touchStartY = 0
let touchCurrentY = 0
let touchStartTime = 0

const handleTouchStart = (e) => {
  touchStartY = e.touches[0].clientY
  touchCurrentY = touchStartY
  touchStartTime = performance.now()
}

const handleTouchMove = (e) => {
  if (!scrollContainer.value) return
  
  touchCurrentY = e.touches[0].clientY
  const deltaY = touchStartY - touchCurrentY
  const container = scrollContainer.value
  const scrollTop = container.scrollTop
  
  if (scrollTop === 0 && deltaY < 0) {
    e.preventDefault()
    const overscroll = Math.abs(deltaY) * 0.3
    container.style.transform = `translateY(${-overscroll}px)`
  } else if (scrollTop >= container.scrollHeight - container.clientHeight && deltaY > 0) {
    e.preventDefault()
    const overscroll = deltaY * 0.3
    container.style.transform = `translateY(${-overscroll}px)`
  }
}

const handleTouchEnd = () => {
  if (!scrollContainer.value) return
  
  const container = scrollContainer.value
  container.style.transform = ''
  
  const elapsed = performance.now() - touchStartTime
  const deltaY = touchStartY - touchCurrentY
  
  if (elapsed < 200 && Math.abs(deltaY) > 50) {
    const velocity = deltaY / elapsed * 1000
    const scrollDistance = velocity * 0.3
    
    const newScrollTop = Math.max(0, Math.min(
      container.scrollTop + scrollDistance,
      container.scrollHeight - container.clientHeight
    ))
    
    animateScrollTo(container, newScrollTop, 300)
  }
}

const animateScrollTo = (container, targetScrollTop, duration) => {
  const start = container.scrollTop
  const startTime = performance.now()
  
  const animate = (currentTime) => {
    const elapsed = currentTime - startTime
    const progress = Math.min(elapsed / duration, 1)
    
    const easeOut = 1 - Math.pow(1 - progress, 3)
    container.scrollTop = start + (targetScrollTop - start) * easeOut
    
    if (progress < 1) {
      requestAnimationFrame(animate)
    }
  }
  
  requestAnimationFrame(animate)
}

onMounted(() => {
  if (scrollContainer.value) {
    scrollContainer.value.addEventListener('touchstart', handleTouchStart, { passive: true })
    scrollContainer.value.addEventListener('touchmove', handleTouchMove, { passive: false })
    scrollContainer.value.addEventListener('touchend', handleTouchEnd, { passive: true })
  }
})

onUnmounted(() => {
  if (scrollContainer.value) {
    scrollContainer.value.removeEventListener('touchstart', handleTouchStart)
    scrollContainer.value.removeEventListener('touchmove', handleTouchMove)
    scrollContainer.value.removeEventListener('touchend', handleTouchEnd)
  }
})
</script>

<style scoped>
.review-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 80px);
  padding: 16px;
  gap: 12px;
  overflow: hidden;
}

.review-container.inline {
  height: auto;
  max-height: calc(100vh - 250px);
  padding: 0;
  overflow: visible;
}

.review-container.inline .review-footer {
  position: sticky;
  bottom: 0;
  background: var(--color-bg-primary);
  padding: 12px 0;
  margin: 0 -16px;
  padding-left: 16px;
  padding-right: 16px;
  box-shadow: var(--shadow-md);
}

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border);
}

.review-header h4 {
  margin: 0;
  font-size: 14px;
  color: var(--color-text-primary);
  font-weight: 600;
}

.header-hint {
  font-size: 11px;
  color: var(--color-text-tertiary);
}

.review-boxes-section {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  position: relative;
  -webkit-overflow-scrolling: touch;
}

.review-boxes-section::-webkit-scrollbar {
  width: 6px;
}

.review-boxes-section::-webkit-scrollbar-track {
  background: var(--color-bg-tertiary);
  border-radius: 3px;
}

.review-boxes-section::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

.review-boxes-section::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-tertiary);
}

.review-boxes-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 20px;
}

.review-box-item {
  width: calc(50% - 6px);
  min-width: 280px;
  background: var(--color-bg-tertiary);
  border: 2px solid var(--color-success);
  border-radius: 10px;
  padding: 12px;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.review-box-item.rejected {
  border-color: var(--color-danger);
  opacity: 0.75;
}

.review-box-item.deleted {
  border-color: var(--color-text-tertiary);
  opacity: 0.5;
  background: var(--color-bg-secondary);
}

.review-box-item.deleted .review-crop {
  filter: grayscale(100%);
}

.review-box-item.has-match {
  border-color: var(--color-primary);
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
}

.review-box-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.box-index {
  font-weight: 600;
  font-size: 13px;
  color: var(--color-text-primary);
}

.box-conf {
  font-size: 11px;
  color: var(--color-text-tertiary);
}

.review-crop-wrapper {
  cursor: pointer;
  position: relative;
  margin-bottom: 10px;
}

.review-crop {
  width: 100%;
  height: 120px;
  object-fit: cover;
  border-radius: 6px;
  transition: transform 0.2s ease;
}

.review-crop-wrapper:hover .review-crop {
  transform: scale(1.02);
}

.click-hint-small {
  position: absolute;
  bottom: 6px;
  right: 6px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 10px;
  pointer-events: none;
}

.review-box-match {
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  padding: 10px;
  margin-bottom: 10px;
}

.match-info {
  display: flex;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.match-label {
  font-size: 12px;
  color: var(--color-text-secondary);
  flex-shrink: 0;
}

.match-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.match-value {
  font-weight: 600;
  font-size: 14px;
  padding: 3px 8px;
  border-radius: var(--radius-sm);
}

.match-value.matched {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.match-name {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-style: italic;
}

.match-value.unmatched {
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

.match-value.low_conf {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.match-conf {
  font-size: 11px;
  color: var(--color-text-tertiary);
}

.sku-input {
  width: 100%;
  padding: 8px 10px;
  font-size: 13px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  margin-bottom: 10px;
}

.sku-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.top5-section {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--color-border);
}

.top5-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  cursor: pointer;
  padding: 4px 0;
  border-radius: 4px;
  transition: background 0.2s ease;
}

.top5-header:hover {
  background: rgba(0, 0, 0, 0.03);
}

.top5-title {
  display: flex;
  align-items: center;
  gap: 6px;
}

.top5-expand-icon {
  font-size: 10px;
  color: var(--color-primary);
  transition: transform 0.2s ease;
}

.top5-title span:last-child {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text-primary);
}

.top5-hint {
  font-size: 10px;
  color: var(--color-primary);
}

.top5-list-container {
  animation: slideDown 0.2s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.top5-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.top5-item {
  width: 100%;
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 8px 10px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.top5-item:hover {
  border-color: var(--color-primary);
  transform: translateY(-2px);
}

.top5-item.selected {
  border-color: var(--color-primary);
  background: rgba(102, 126, 234, 0.1);
}

.top5-item.top1 {
  border-color: var(--color-primary);
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(102, 126, 234, 0.1) 100%);
}

.top5-rank {
  font-size: 14px;
  margin-right: 6px;
  flex-shrink: 0;
}

.top5-image {
  width: 40px;
  height: 40px;
  object-fit: cover;
  border-radius: 4px;
  margin-right: 8px;
  flex-shrink: 0;
  max-width: 40px;
}

.top5-info {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  flex: 1;
  min-width: 80px;
  max-width: 200px;
  gap: 2px;
}

.top5-sku {
  font-size: 14px;
  color: var(--color-text-primary);
  font-weight: 600;
}

.top5-name {
  font-size: 12px;
  color: var(--color-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.top5-sim {
  font-size: 13px;
  color: var(--color-success);
  font-weight: 600;
  margin-left: auto;
  margin-right: 8px;
  flex-shrink: 0;
}

.review-box-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 10px;
  border-top: 1px solid var(--color-border);
}

.modified-indicator {
  flex: 1;
}

.modified-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
  font-size: 11px;
  font-weight: 500;
  border-radius: 10px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.btn-delete {
  padding: 6px 12px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
}

.btn-delete:hover {
  border-color: var(--color-danger);
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

.sku-input.modified {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
  background: rgba(102, 126, 234, 0.03);
}

.review-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}

.btn {
  padding: 10px 24px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-primary {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn-default {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

.btn-default:hover {
  background: var(--color-bg-secondary);
}

@media (max-width: 768px) {
  .review-container {
    height: calc(100vh - 60px);
    padding: 12px;
  }
  
  .review-box-item {
    width: calc(50% - 6px);
  }
  
  .top5-item {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .top5-item {
    width: 100%;
  }
}
</style>
