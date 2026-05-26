<template>
  <div>
    <div class="batch-result-header">
      <span class="result-index">{{ index + 1 }}</span>
      <span class="result-filename">{{ result.fileName }}</span>
      <span v-if="result.success" class="result-status success">✓ 成功 ({{ result.count || 0 }}个)</span>
      <span v-else class="result-status error">✗ 失败</span>
    </div>

    <div v-if="result.success" class="batch-result-content">
      <div class="result-images">
        <div class="result-image-box">
          <img v-if="result.image_with_boxes" :src="'data:image/jpeg;base64,' + result.image_with_boxes" class="batch-result-image">
        </div>

        <div v-if="result.boxes && result.boxes.length > 0" class="crops-preview">
          <div v-for="(box, boxIdx) in result.boxes.slice(0, 6)" :key="boxIdx" class="crop-item">
            <img v-if="result.crops && result.crops[boxIdx]" :src="'data:image/jpeg;base64,' + result.crops[boxIdx]" class="crop-thumb">
            <div class="crop-info">
              <span class="crop-conf">{{ (box.confidence * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <div class="result-actions">
        <button class="btn-small btn-primary" @click="handleReview">审核检测结果</button>
      </div>
    </div>

    <div v-else class="batch-result-error">
      <span>{{ result.error }}</span>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  result: {
    type: Object,
    required: true
  },
  index: {
    type: Number,
    required: true
  }
})

const emit = defineEmits(['review'])

const handleReview = () => {
  emit('review', props.result, props.index)
}
</script>

<style scoped>
.batch-result-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 15px;
  background: #fafafa;
  border-bottom: 1px solid #f0f0f0;
}

.result-index {
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

.result-filename {
  flex: 1;
  font-weight: 500;
  color: #333;
}

.result-status {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.result-status.success {
  background: #e1f3d8;
  color: #67c23a;
}

.result-status.error {
  background: #fef0f0;
  color: #f56c6c;
}

.batch-result-content {
  padding: 15px;
}

.result-images {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  align-items: flex-start;
}

.result-image-box {
  flex: 1;
  min-width: 200px;
  max-width: 400px;
}

.batch-result-image {
  width: 100%;
  max-height: 200px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.crops-preview {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.crop-item {
  width: 80px;
  text-align: center;
}

.crop-thumb {
  width: 80px;
  height: 80px;
  object-fit: cover;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.crop-info {
  margin-top: 5px;
}

.crop-conf {
  font-size: 11px;
  color: #666;
}

.result-actions {
  margin-top: 15px;
}

.btn-small {
  padding: 6px 16px;
  border: none;
  border-radius: 4px;
  font-size: 13px;
  cursor: pointer;
}

.btn-small.btn-primary {
  background: #409eff;
  color: white;
}

.batch-result-error {
  padding: 15px;
  background: #fef0f0;
  color: #f56c6c;
  font-size: 14px;
}
</style>
