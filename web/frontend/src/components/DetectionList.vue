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
</style>