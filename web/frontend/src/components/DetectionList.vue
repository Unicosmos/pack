<template>
  <div class="detection-list">
    <div v-if="!items || items.length === 0" class="empty-state">
      <el-icon class="empty-icon"><Document /></el-icon>
      <p>暂无检测结果</p>
    </div>
    <div v-else class="list-container">
      <div v-for="(item, idx) in items" :key="idx" class="detection-item">
        <div class="item-thumb">
          <img
            v-if="item.crop"
            :src="item.crop"
            :alt="`Box ${idx + 1}`"
          />
          <div v-else class="thumb-placeholder">N/A</div>
        </div>

        <div class="item-info">
          <div class="item-header">
            <strong>箱体 {{ idx + 1 }}</strong>
            <span class="item-conf">置信度: {{ (item.confidence * 100).toFixed(1) }}%</span>
          </div>
          <div class="item-bbox">位置: [{{ item.bbox.join(', ') }}]</div>
        </div>

        <div class="item-match">
          <el-tag
            v-if="item.match"
            :type="getMatchTagType(item.match.status)"
            size="large"
          >
            {{ item.match.sku_id || '未知' }}
            <span v-if="item.match.similarity" class="match-sim">
              ({{ (item.match.similarity * 100).toFixed(1) }}%)
            </span>
          </el-tag>
          <el-tag v-else type="info" size="large">待匹配</el-tag>
        </div>
      </div>

      <div v-if="items.length > 0 && items[0].match && items[0].match.top5_labels && items[0].match.top5_labels.length > 0" class="top5-section">
        <h4>Top-5 匹配结果</h4>
        <div class="top5-grid">
          <div
            v-for="(label, labelIdx) in items[0].match.top5_labels"
            :key="labelIdx"
            class="top5-item"
            :class="{ 'top1': labelIdx === 0 }"
          >
            <div class="top5-thumb">
              <img
                v-if="label.image_name"
                :src="getSkuImageUrl(label.image_name)"
                :alt="label.sku_name || label.label"
              />
              <div v-else class="top5-placeholder">
                <span class="top5-rank">{{ labelIdx + 1 }}</span>
              </div>
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
</template>

<script setup>
import { Document } from '@element-plus/icons-vue'

defineProps({
  items: {
    type: Array,
    default: () => []
  }
})

const getMatchTagType = (status) => {
  switch (status) {
    case 'matched': return 'success'
    case 'unmatched': return 'warning'
    case 'low_conf': return 'warning'
    default: return 'info'
  }
}

const getSkuImageUrl = (imageName) => {
  return `/static/sku_images/${imageName}`
}
</script>

<style scoped>
.detection-list {
  width: 100%;
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.list-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.item-thumb {
  width: 80px;
  height: 80px;
  border-radius: 6px;
  overflow: hidden;
  background: #f0f0f0;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.item-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb-placeholder {
  color: #999;
  font-size: 12px;
}

.item-info {
  flex: 1;
  min-width: 0;
}

.item-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 4px;
}

.item-conf {
  font-size: 13px;
  color: #666;
}

.item-bbox {
  font-size: 12px;
  color: #999;
}

.item-match {
  flex-shrink: 0;
}

.match-sim {
  margin-left: 4px;
  opacity: 0.8;
}

.top5-section {
  margin-top: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
}

.top5-section h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.top5-grid {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.top5-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px;
  background: #fff;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  width: calc(20% - 10px);
  min-width: 120px;
}

.top5-item.top1 {
  border-color: #67c23a;
  background: #f0f9eb;
}

.top5-thumb {
  width: 80px;
  height: 80px;
  border-radius: 4px;
  overflow: hidden;
  background: #f0f0f0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
}

.top5-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.top5-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e0e0e0;
}

.top5-rank {
  font-size: 24px;
  font-weight: bold;
  color: #999;
}

.top5-info {
  text-align: center;
  width: 100%;
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
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.top5-similarity {
  font-size: 11px;
  color: #999;
}

@media (max-width: 768px) {
  .top5-item {
    width: calc(33.33% - 8px);
  }
}
</style>