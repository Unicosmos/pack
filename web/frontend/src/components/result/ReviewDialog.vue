<template>
  <div v-if="task" class="review-container">
    <div class="review-header">
      <h4>检测到的箱体 ({{ boxes.length }}个)</h4>
      <span class="header-hint">💡 点击箱体图片可查看大图，点击Top5结果可选择匹配</span>
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
          
          <div class="review-crop-wrapper" @click="openLargeImage(box.crop_base64 ? { url: 'data:image/jpeg;base64,' + box.crop_base64 } : '')">
            <SkuImage
              :image-path="box.crop_base64 ? { url: 'data:image/jpeg;base64,' + box.crop_base64 } : ''"
              height="120px"
              class="review-crop"
            />
            <div class="click-hint-small">👆 查看大图</div>
          </div>

          <div v-if="box.match_result" class="review-box-match">
            <div class="match-info">
              <span class="match-label">匹配结果:</span>
              <span class="match-value" :class="box.match_result.status">
                {{ box.match_result.sku_id || '未匹配' }}
              </span>
              <span v-if="box.match_result.similarity" class="match-conf">
                相似度: {{ (box.match_result.similarity * 100).toFixed(1) }}%
              </span>
            </div>
            
            <input
              type="text"
              class="sku-input"
              v-model="box.custom_sku"
              placeholder="输入自定义SKU"
              title="修改匹配SKU"
            />

            <div v-if="box.match_result.top5_labels && box.match_result.top5_labels.length > 0" class="top5-section">
              <div class="top5-header">
                <span>Top 5 匹配结果</span>
                <span class="top5-hint">点击选择</span>
              </div>
              <div class="top5-list">
                <div
                  v-for="(label, lIdx) in box.match_result.top5_labels.slice(0, 5)"
                  :key="lIdx"
                  class="top5-item"
                  :class="{ 'selected': box.custom_sku === label.sku_id }"
                  @click="selectTop5Sku(box, label)"
                >
                  <SkuImage
                    :image-path="label.image_path ? getSkuImageUrl(label.image_path) : ''"
                    height="60px"
                    class="top5-image"
                    :lazy="true"
                  />
                  <div class="top5-info">
                    <span class="top5-sku">{{ label.sku_id || label.label }}</span>
                    <span class="top5-sim">{{ (label.similarity * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="review-box-actions">
            <button
              class="btn-approve"
              :class="{ active: box.status === 'approved' }"
              @click="setBoxStatus(box, 'approved')"
            >
              ✓ 批准
            </button>
            <button
              class="btn-reject"
              :class="{ active: box.status === 'rejected' }"
              @click="setBoxStatus(box, 'rejected')"
            >
              ✗ 拒绝
            </button>
            <button
              class="btn-delete"
              :class="{ active: box.status === 'deleted' }"
              @click="setBoxStatus(box, 'deleted')"
              title="删除此框"
            >
              🗑️
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="review-footer">
      <button class="btn btn-default" @click="handleCancel">取消</button>
      <button class="btn btn-primary" @click="handleSubmit">提交审核</button>
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
import SkuImage from './SkuImage.vue'
import ImageViewer from './ImageViewer.vue'
import { getImageUrlFromPath } from '../../api/client'

const props = defineProps({
  task: { type: Object, default: null }
})

const emit = defineEmits(['cancel', 'submit'])

const boxes = ref([])
const scrollContainer = ref(null)
const isScrolling = ref(false)
const scrollStartY = ref(0)
const scrollCurrentY = ref(0)

const showImageViewer = ref(false)
const currentImageUrl = ref('')
const currentImageName = ref('')

const getTaskImageUrl = () => {
  if (!props.task) return ''
  return `/api/tasks/${props.task.id}/image`
}

const getSkuImageUrl = (imagePath) => {
  return getImageUrlFromPath(imagePath)
}

const initData = () => {
  if (!props.task) return
  
  boxes.value = []
  
  const detections = props.task.result?.detections
  if (detections?.boxes) {
    boxes.value = detections.boxes.map(box => ({
      ...box,
      status: box.status || 'approved',
      custom_sku: ''
    }))
  } else if (props.task.result?.boxes) {
    boxes.value = props.task.result.boxes.map((box, idx) => ({
      ...box,
      box_id: String(idx),
      status: box.status || 'approved',
      custom_sku: ''
    }))
  }

  if (props.task.result?.matches) {
    const matches = props.task.result.matches
    boxes.value.forEach(box => {
      const matchKey = typeof box.box_id === 'number' ? `box_${box.box_id}` : box.box_id
      if (matches[matchKey]) {
        box.match_result = matches[matchKey]
      } else if (matches[parseInt(box.box_id)]) {
        box.match_result = matches[parseInt(box.box_id)]
      }
    })
  }
}

watch(() => props.task, (task) => {
  if (task) {
    initData()
  }
}, { immediate: true })

const setBoxStatus = (box, status) => {
  box.status = status
}

const selectTop5Sku = (box, label) => {
  box.custom_sku = label.sku_id || label.label
  if (box.match_result) {
    box.match_result.sku_id = box.custom_sku
  }
  ElMessage.success(`已选择 SKU: ${box.custom_sku}`)
}

const handleCancel = () => {
  emit('cancel')
}

const handleSubmit = () => {
  const nonDeletedBoxes = boxes.value.filter(b => b.status !== 'deleted')
  const approved = nonDeletedBoxes.filter(b => b.status === 'approved').length
  const rejected = nonDeletedBoxes.filter(b => b.status === 'rejected').length
  const deleted = boxes.value.length - nonDeletedBoxes.length

  const resultBoxes = nonDeletedBoxes.map((box, idx) => ({
    ...box,
    box_id: `box_${idx}`
  }))

  emit('submit', {
    task: props.task,
    boxes: resultBoxes,
    approvedCount: approved,
    rejectedCount: rejected,
    deletedCount: deleted
  })

  let msg = `审核完成: 通过 ${approved} 个`
  if (rejected > 0) msg += `，拒绝 ${rejected} 个`
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

const handleWheel = (e) => {
  if (!scrollContainer.value) return
  
  const container = scrollContainer.value
  const scrollTop = container.scrollTop
  const scrollHeight = container.scrollHeight
  const clientHeight = container.clientHeight
  const deltaY = e.deltaY

  if ((scrollTop === 0 && deltaY < 0) || (scrollTop >= scrollHeight - clientHeight - 1 && deltaY > 0)) {
    e.preventDefault()
    
    if (deltaY < 0) {
      animateScroll(container, -50, 0, 200)
    } else {
      animateScroll(container, 50, scrollHeight - clientHeight, 200)
    }
  }
}

const animateScroll = (container, offset, limit, duration) => {
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

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid #e5e5e5;
}

.review-header h4 {
  margin: 0;
  font-size: 14px;
  color: #333;
  font-weight: 600;
}

.header-hint {
  font-size: 11px;
  color: #999;
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
  background: #f1f1f1;
  border-radius: 3px;
}

.review-boxes-section::-webkit-scrollbar-thumb {
  background: #c0c0c0;
  border-radius: 3px;
}

.review-boxes-section::-webkit-scrollbar-thumb:hover {
  background: #a0a0a0;
}

.review-boxes-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 20px;
}

.review-box-item {
  width: calc(50% - 6px);
  background: #fafafa;
  border: 2px solid #67c23a;
  border-radius: 10px;
  padding: 12px;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.review-box-item.rejected {
  border-color: #f56c6c;
  opacity: 0.75;
}

.review-box-item.deleted {
  border-color: #909399;
  opacity: 0.5;
  background: #f5f5f5;
}

.review-box-item.deleted .review-crop {
  filter: grayscale(100%);
}

.review-box-item.has-match {
  border-color: #667eea;
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
  color: #333;
}

.box-conf {
  font-size: 11px;
  color: #999;
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
  background: #f8fafc;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 10px;
}

.match-info {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.match-label {
  font-size: 12px;
  color: #666;
}

.match-value {
  font-weight: 600;
  font-size: 13px;
  color: #67c23a;
}

.match-value.unmatched {
  color: #f56c6c;
}

.match-value.low_conf {
  color: #e6a23c;
}

.match-conf {
  font-size: 11px;
  color: #999;
}

.sku-input {
  width: 100%;
  padding: 8px 10px;
  font-size: 13px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  background: white;
  margin-bottom: 10px;
}

.sku-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.top5-section {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed #e0e0e0;
}

.top5-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.top5-header span:first-child {
  font-size: 12px;
  font-weight: 500;
  color: #333;
}

.top5-hint {
  font-size: 10px;
  color: #667eea;
}

.top5-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.top5-item {
  width: calc(50% - 3px);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.top5-item:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.top5-item.selected {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.1);
}

.top5-image {
  width: 100%;
  height: 60px;
  object-fit: cover;
  border-radius: 4px;
  margin-bottom: 4px;
}

.top5-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.top5-sku {
  font-size: 10px;
  color: #333;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 70%;
}

.top5-sim {
  font-size: 10px;
  color: #67c23a;
  font-weight: 600;
}

.review-box-actions {
  display: flex;
  gap: 8px;
}

.btn-approve,
.btn-reject {
  flex: 1;
  padding: 7px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-approve {
  background: #e1f3d8;
  color: #67c23a;
}

.btn-approve.active,
.btn-approve:hover {
  background: #67c23a;
  color: white;
}

.btn-reject {
  background: #fef0f0;
  color: #f56c6c;
}

.btn-reject.active,
.btn-reject:hover {
  background: #f56c6c;
  color: white;
}

.btn-delete {
  padding: 7px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
  background: #f5f5f5;
  color: #909399;
}

.btn-delete.active,
.btn-delete:hover {
  background: #f56c6c;
  color: white;
}

.review-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px solid #e5e5e5;
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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn-default {
  background: #f5f5f5;
  color: #666;
}

.btn-default:hover {
  background: #e8e8e8;
}

@media (max-width: 768px) {
  .review-container {
    height: calc(100vh - 60px);
    padding: 12px;
  }
  
  .review-box-item {
    width: 100%;
  }
  
  .top5-item {
    width: calc(33.33% - 4px);
  }
}

@media (max-width: 480px) {
  .top5-item {
    width: calc(50% - 3px);
  }
}
</style>
