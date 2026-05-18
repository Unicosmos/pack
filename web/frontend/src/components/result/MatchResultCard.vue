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
            <MatchTags :match="result.matches && result.matches[boxIdx]" />
            <MatchTop5 v-if="result.matches && result.matches[boxIdx] && result.matches[boxIdx].top5_labels" :labels="result.matches[boxIdx].top5_labels" :status="result.matches[boxIdx].status" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import MatchTags from './MatchTags.vue'
import MatchTop5 from './MatchTop5.vue'

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
  background: #667eea;
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
</style>
