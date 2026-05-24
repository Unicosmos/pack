<template>
  <div v-if="match" class="match-tags">
    <span :class="['match-tag', match.status]">
      {{ match.sku_id || '未匹配' }}
      <span v-if="match.similarity">
        ({{ (match.similarity * 100).toFixed(1) }}%)
      </span>
    </span>
  </div>
</template>

<script setup>
defineProps({
  match: {
    type: Object,
    default: null
  }
})
</script>

<style scoped>
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
</style>
