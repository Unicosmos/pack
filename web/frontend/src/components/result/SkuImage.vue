<template>
  <div class="sku-image" :style="{ width: width, height: height }">
    <img
      v-if="src"
      :src="src"
      :alt="alt"
      @error="handleError"
      :class="{ 'error': hasError }"
      @load="onLoad"
    />
    <div v-else class="placeholder">
      <span class="placeholder-icon">{{ placeholderIcon }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { getImageUrlFromPath } from '@/api/client'

const props = defineProps({
  imagePath: {
    type: [String, Object],
    default: ''
  },
  width: {
    type: String,
    default: '100%'
  },
  height: {
    type: String,
    default: '80px'
  },
  alt: {
    type: String,
    default: ''
  },
  placeholderIcon: {
    type: String,
    default: '📷'
  }
})

const hasError = ref(false)

const src = computed(() => {
  if (!props.imagePath) return ''
  
  // 如果是直接URL字符串（以 / 或 http 开头），直接返回
  if (typeof props.imagePath === 'string') {
    if (props.imagePath.startsWith('/') || props.imagePath.startsWith('http')) {
      return props.imagePath
    }
  }
  
  return getImageUrlFromPath(props.imagePath)
})

function handleError() {
  console.error('[SkuImage] 图片加载失败:', props.imagePath)
  hasError.value = true
}

function onLoad() {
  // 图片加载成功
}
</script>

<style scoped>
.sku-image {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f5f5f5;
  overflow: hidden;
}

.sku-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.sku-image img.error {
  display: none;
}

.placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
  font-size: 24px;
}
</style>
