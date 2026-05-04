<template>
  <div class="stats-card" :style="{ borderLeftColor: color }">
    <div class="stats-number" :style="{ color: color }">{{ value }}</div>
    <div class="stats-label">{{ label }}</div>
  </div>
</template>

<script setup>
defineProps({
  value: {
    type: [Number, String],
    required: true
  },
  label: {
    type: String,
    required: true
  },
  color: {
    type: String,
    default: '#667eea'
  }
})
</script>

<style scoped>
.stats-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  border-left: 4px solid;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  text-align: center;
}

.stats-number {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.2;
}

.stats-label {
  font-size: 14px;
  color: #666;
  margin-top: 8px;
}
</style>