<template>
  <div class="detection-list">
    <div v-if="mode === 'review'" class="batch-results-list">
      <div v-for="(result, idx) in results" :key="idx" class="batch-result-item">
        <BatchResultCard :result="result" :index="idx" @review="handleReview" />
      </div>
    </div>

    <div v-else class="match-results-container">
      <div v-for="(result, idx) in results" :key="idx" class="match-result-card">
        <MatchResult :result="result" :index="idx" />
      </div>
    </div>
  </div>
</template>

<script setup>
import BatchResultCard from '@task/BatchResultCard.vue'
import MatchResult from '@task/MatchResult.vue'

const props = defineProps({
  results: {
    type: Array,
    default: () => []
  },
  mode: {
    type: String,
    default: 'review'
  }
})

const emit = defineEmits(['review'])

const handleReview = (result, idx) => {
  emit('review', result, idx)
}
</script>

<style scoped>
.detection-list {
  margin-top: 20px;
}

.batch-results-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.batch-result-item {
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
}

.match-results-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.match-result-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}
</style>
