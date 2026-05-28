<template>
  <div>
    <div class="match-result-header">
      <span class="match-result-index">{{ index + 1 }}</span>
      <span class="match-result-filename">{{ result.fileName }}</span>
    </div>

    <div v-if="result.success" class="match-result-content">
      <div class="match-stats">
        <div class="stat-item">
          <div class="stat-num">{{ result.count || 0 }}</div>
          <div class="stat-label">检测数量</div>
        </div>
        <div class="stat-item success">
          <div class="stat-num">{{ result.matched_count || 0 }}</div>
          <div class="stat-label">已匹配</div>
        </div>
        <div class="stat-item warning">
          <div class="stat-num">{{ result.unmatched_count || 0 }}</div>
          <div class="stat-label">未匹配</div>
        </div>
      </div>

      <div class="match-images">
        <div class="match-image-box">
          <img v-if="result.image_with_boxes" :src="'data:image/jpeg;base64,' + result.image_with_boxes" class="match-result-image">
        </div>
      </div>

      <div v-if="result.boxes && result.boxes.length > 0" class="match-boxes-list">
        <div v-for="(box, boxIdx) in result.boxes" :key="boxIdx" class="match-box-item">
          <img v-if="result.crops && result.crops[boxIdx]" :src="'data:image/jpeg;base64,' + result.crops[boxIdx]" class="match-box-thumb">
          <div class="match-box-info">
            <div class="match-box-header">
              <strong>箱体 {{ boxIdx + 1 }}</strong>
              <span class="box-conf">{{ (box.confidence * 100).toFixed(1) }}%</span>
            </div>
            <div v-if="result.matches && result.matches[boxIdx]" class="match-tags">
              <span :class="['match-tag', result.matches[boxIdx].status]">
                {{ result.matches[boxIdx].sku_id || '未匹配' }}
                <span v-if="result.matches[boxIdx].similarity">
                  ({{ (result.matches[boxIdx].similarity * 100).toFixed(1) }}%)
                </span>
              </span>
            </div>
            <div v-if="result.matches && result.matches[boxIdx] && result.matches[boxIdx].top5_labels" class="match-top5">
              <div class="top5-header">
                <span class="top5-title">Top-5 候选</span>
                <span :class="['top5-status', result.matches[boxIdx].status]">
                  {{ getStatusText(result.matches[boxIdx].status) }}
                </span>
              </div>
              <div class="top5-grid">
                <div
                  v-for="(label, labelIdx) in result.matches[boxIdx].top5_labels"
                  :key="labelIdx"
                  class="top5-item"
                  :class="{ 'top1': labelIdx === 0, 'selected': labelIdx === 0 && result.matches[boxIdx].status === 'matched' }"
                >
                  <div class="top5-thumb">
                    <SkuImage
                      :image-path="label.image_path"
                      :alt="label.sku_name || label.label"
                      :placeholder-icon="String(labelIdx + 1)"
                    />
                  </div>
                  <div class="top5-info">
                    <div class="top5-sku-id">{{ label.sku_id || label.label }}</div>
                    <div class="top5-sku-name">{{ label.sku_name }}</div>
                    <div class="top5-similarity">相似度: {{ (label.similarity * 100).toFixed(1) }}%</div>
                  </div>
                </div>
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

defineProps({
  result: {
    type: Object,
    required: true
  },
  index: {
    type: Number,
    required: true
  }
})

const getStatusText = (status) => {
  switch (status) {
    case 'matched': return '✓ 已匹配'
    case 'low_conf': return '⚠️ 低置信'
    case 'unmatched': return '✗ 未匹配'
    default: return ''
  }
}
</script>

<style scoped>
.match-result-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 15px;
  background: #fafafa;
  border-bottom: 1px solid #e0e0e0;
}

.match-result-index {
  width: 24px;
  height: 24px;
  background: #0066cc;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
}

.match-result-filename {
  font-weight: 500;
  color: #333;
}

.match-result-content {
  padding: 15px;
}

.match-stats {
  display: flex;
  gap: 20px;
  margin-bottom: 15px;
}

.stat-item {
  flex: 1;
  padding: 15px;
  background: #f5f5f5;
  border-radius: 8px;
  text-align: center;
}

.stat-item.success {
  background: #e1f3d8;
}

.stat-item.warning {
  background: #faecd8;
}

.stat-num {
  font-size: 24px;
  font-weight: 600;
  color: #333;
}

.stat-label {
  font-size: 12px;
  color: #666;
  margin-top: 5px;
}

.match-images {
  margin-bottom: 15px;
}

.match-image-box {
  max-width: 500px;
}

.match-result-image {
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.match-boxes-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.match-box-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 10px;
  background: #fafafa;
  border-radius: 8px;
}

.match-box-thumb {
  width: 80px;
  height: 80px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
}

.match-box-info {
  flex: 1;
}

.match-box-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}

.box-conf {
  font-size: 12px;
  color: #666;
}

.match-tags {
  margin-top: 8px;
}

.match-tag {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
}

.match-tag.matched {
  background: #e1f3d8;
  color: #67c23a;
}

.match-tag.low_conf {
  background: #faecd8;
  color: #e6a23c;
}

.match-tag.unmatched {
  background: #fef0f0;
  color: #f56c6c;
}

.match-top5 {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #e0e0e0;
}

.top5-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.top5-title {
  font-size: 13px;
  font-weight: 500;
  color: #666;
}

.top5-status {
  font-size: 12px;
  padding: 3px 8px;
  border-radius: 3px;
}

.top5-status.matched {
  background: #e1f3d8;
  color: #67c23a;
}

.top5-status.low_conf {
  background: #faecd8;
  color: #e6a23c;
}

.top5-status.unmatched {
  background: #fef0f0;
  color: #f56c6c;
}

.top5-grid {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.top5-item {
  width: calc(20% - 8px);
  min-width: 100px;
  background: #fafafa;
  border-radius: 6px;
  overflow: hidden;
  border: 2px solid transparent;
  transition: all 0.3s;
}

.top5-item.top1 {
  border-color: #0066cc;
  background: #f0f5ff;
}

.top5-item.selected {
  border-color: #67c23a;
  background: #f0f9eb;
}

.top5-thumb {
  width: 100%;
  height: 80px;
}

.top5-info {
  padding: 8px;
}

.top5-sku-id {
  font-size: 12px;
  font-weight: 600;
  color: #333;
  margin-bottom: 2px;
}

.top5-sku-name {
  font-size: 11px;
  color: #666;
  margin-bottom: 3px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.top5-similarity {
  font-size: 11px;
  color: #999;
}
</style>
