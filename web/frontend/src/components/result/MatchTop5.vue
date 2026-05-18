<template>
  <div v-if="labels && labels.length > 0" class="match-top5">
    <div class="top5-header">
      <span class="top5-title">Top-5 候选</span>
      <span :class="['top5-status', status]">
        {{ statusText }}
      </span>
    </div>
    <div class="top5-grid">
      <div
        v-for="(label, labelIdx) in labels"
        :key="labelIdx"
        class="top5-item"
        :class="{ 'top1': labelIdx === 0, 'selected': labelIdx === 0 && status === 'matched' }"
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
</template>

<script setup>
import { computed } from 'vue'
import SkuImage from './SkuImage.vue'

const props = defineProps({
  labels: {
    type: Array,
    default: () => []
  },
  status: {
    type: String,
    default: ''
  }
})

const statusText = computed(() => {
  switch (props.status) {
    case 'matched': return '✓ 已匹配'
    case 'low_conf': return '⚠️ 低置信'
    case 'unmatched': return '✗ 未匹配'
    default: return ''
  }
})
</script>

<style scoped>
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
  border-color: #667eea;
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
