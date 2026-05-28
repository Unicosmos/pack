<template>
  <div class="task-detail-panel">
    <template v-if="!hideInfo">
    <div class="detail-actions" v-if="showActions">
      <button v-if="shouldShowDetect(task)" class="btn btn-primary" @click="$emit('detect', task)">识别</button>
    </div>

    <div class="detail-grid">
      <div class="detail-item">
        <span class="detail-label">图片名称：</span>
        <span class="detail-value">{{ task.image_name }}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">状态：</span>
        <span :class="['detail-value', 'status-badge', task.status]">{{ getStatusText(task.status) }}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">检测数量：</span>
        <span class="detail-value">{{ task.box_count || 0 }}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">匹配数量：</span>
        <span class="detail-value">{{ task.matched_count || 0 }}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">未匹配数量：</span>
        <span class="detail-value">{{ task.unmatched_count || 0 }}</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">创建时间：</span>
        <span class="detail-value">{{ formatDate(task.created_at) }}</span>
      </div>
      <div class="detail-item" v-if="task.completed_at">
        <span class="detail-label">完成时间：</span>
        <span class="detail-value">{{ formatDate(task.completed_at) }}</span>
      </div>
    </div>
    </template>

    <div class="result-section">
      <div class="preview-row">
        <div class="preview-section" @click="$emit('view-image', getTaskImagePath(task), task.image_name)">
          <div class="preview-title">
            <span>原图</span>
            <span class="click-indicator">👆 点击查看大图</span>
          </div>
          <div class="preview-img-wrapper">
            <SkuImage
              :image-path="getTaskImagePath(task)"
              fit="contain"
              class="preview-img clickable"
            />
          </div>
        </div>
        <div v-if="getTaskPreviewPath(task)" class="preview-section" @click="$emit('view-image', getTaskPreviewPath(task).url, task.image_name + ' (检测结果)')">
          <div class="preview-title">
            <span>检测结果（带框）</span>
            <span class="click-indicator">👆 点击查看大图</span>
          </div>
          <div class="preview-img-wrapper">
            <SkuImage
              :image-path="getTaskPreviewPath(task)"
              fit="contain"
              class="preview-img clickable"
            />
          </div>
        </div>
      </div>

      <div v-if="getDetectionBoxes(task).length > 0" class="detection-boxes-preview">
        <h5>
          识别结果 ({{ getDetectionBoxes(task).length }}个)
          <span class="box-hint">👆 点击箱体可查看/修改匹配</span>
        </h5>
        <div class="boxes-grid">
          <div v-for="(box, idx) in getDetectionBoxes(task)" :key="box.box_id" class="box-item" :class="{ deleted: box.status === 'deleted' }">
            <div v-if="box.status !== 'deleted'" class="box-main">
              <SkuImage
                :image-path="getBoxImageUrl(box)"
                :placeholder-icon="String(idx + 1)"
                height="80px"
                fit="contain"
                class="clickable"
                @click="$emit('match-box', { box, index: idx })"
              />
              <div class="box-info">
                <span class="box-idx">箱体 {{ idx + 1 }}</span>
                <span class="box-conf">置信度: {{ (box.confidence * 100).toFixed(1) }}%</span>
                <span v-if="getMatchResultForTask(task, box.box_id)" class="box-match" :class="getMatchResultForTask(task, box.box_id).status">
                  {{ getMatchResultForTask(task, box.box_id).sku_id || '未匹配' }}
                </span>
              </div>
              <button class="btn-delete-box" @click.stop="$emit('delete-box', { box, index: idx })" title="删除此箱体">
                🗑️
              </button>
            </div>
            <div v-else class="box-main box-deleted">
              <SkuImage
                :image-path="getBoxImageUrl(box)"
                :placeholder-icon="String(idx + 1)"
                height="80px"
                fit="contain"
              />
              <div class="box-info">
                <span class="box-idx">箱体 {{ idx + 1 }}</span>
                <span class="box-deleted-text">已删除</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import SkuImage from '@sku/SkuImage.vue'
import {
  getStatusText,
  getStatusBadgeClass,
  formatDate,
  shouldShowDetect
} from '@utils/taskUtils'
import { getImageUrlFromPath } from '@api/client'

defineProps({
  task: { type: Object, required: true },
  imageHeight: { type: String, default: '200px' },
  showActions: { type: Boolean, default: false },
  hideInfo: { type: Boolean, default: false }
})

defineEmits(['view-image', 'detect', 'match-box', 'delete-box'])

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
</script>

<style scoped>
.task-detail-panel {
  padding: var(--spacing-lg);
}

.detail-actions {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--color-border-light);
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.detail-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.detail-value {
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
}

.result-section h5 {
  margin: var(--spacing-lg) 0 var(--spacing-md) 0;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.box-hint {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-weight: normal;
}

.preview-row {
  display: flex;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  max-height: 250px;
}

.preview-section {
  flex: 1;
  cursor: pointer;
  overflow: hidden;
}

.preview-img-wrapper {
  width: 100%;
  height: 200px;
  overflow: hidden;
  background: var(--color-bg-secondary);
  border-radius: var(--radius-md);
}

.preview-img-wrapper :deep(.sku-image) {
  height: 100% !important;
  width: 100%;
  background: transparent;
}

.preview-img-wrapper :deep(.sku-image img) {
  object-fit: contain;
  max-height: 200px;
}

.preview-title {
  margin-bottom: var(--spacing-sm);
  font-size: var(--font-size-base);
  font-weight: 500;
  color: var(--color-text-secondary);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.click-indicator {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  font-weight: normal;
}

.preview-img {
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}

.preview-img.clickable,
.box-item .clickable {
  cursor: pointer;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}

.preview-img.clickable:hover,
.box-item:hover .clickable {
  transform: scale(1.02);
  box-shadow: var(--shadow-lg);
}

.boxes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
  max-width: 100%;
}

.box-item {
  min-width: 200px;
  max-width: 300px;
  justify-self: start;
  position: relative;
}

.box-item.deleted {
  opacity: 0.5;
  filter: grayscale(100%);
}

.box-main {
  position: relative;
  cursor: pointer;
  transition: transform var(--transition-fast);
  padding: 8px;
  border-radius: 8px;
}

.box-main:hover {
  transform: translateY(-2px);
  background: var(--color-bg-tertiary);
}

.box-main.box-deleted {
  cursor: default;
  transform: none;
  background: none;
}

.btn-delete-box {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 28px;
  height: 28px;
  border: none;
  background: rgba(245, 108, 108, 0.9);
  color: white;
  border-radius: 50%;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  z-index: 10;
  opacity: 0;
}

.box-main:hover .btn-delete-box {
  opacity: 1;
}

.btn-delete-box:hover {
  background: rgba(220, 53, 69, 1);
  transform: scale(1.1);
}

.box-deleted-text {
  color: var(--color-text-secondary);
  font-size: 12px;
}

.box-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xs);
  font-size: var(--font-size-xs);
}

.box-idx {
  font-weight: 600;
  color: var(--color-primary);
  font-size: var(--font-size-sm);
}

.box-conf {
  color: var(--color-text-secondary);
}

.box-match {
  padding: 3px 6px;
  border-radius: var(--radius-sm);
  font-size: var(--font-size-xs);
  text-align: center;
}

.box-match.matched {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.box-match.low_conf {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.box-match.unmatched {
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

@media (max-width: 900px) {
  .detail-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .boxes-grid {
    grid-template-columns: repeat(2, minmax(150px, 1fr));
  }

  .box-item {
    width: auto;
  }
}

@media (max-width: 768px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .boxes-grid {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }

  .box-item {
    width: auto;
  }
}
</style>
