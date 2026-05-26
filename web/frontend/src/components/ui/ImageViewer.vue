<template>
  <Teleport to="body">
    <div v-if="visible" class="image-viewer-overlay" @click="close">
      <div class="image-viewer-container" @click.stop>
        <div class="image-viewer-header">
          <span class="image-name">{{ imageName }}</span>
          <button class="close-btn" @click="close">×</button>
        </div>
        <div class="image-viewer-wrapper">
          <img :src="imageUrl" :alt="imageName" @error="handleError" />
        </div>
        <div class="image-viewer-footer">
          <span class="image-info">{{ imageInfo }}</span>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { watch } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  imageUrl: { type: String, default: '' },
  imageName: { type: String, default: '' },
  imageInfo: { type: String, default: '' }
})

const emit = defineEmits(['update:visible'])

watch(() => props.visible, (val) => {
  if (val) {
    document.body.style.overflow = 'hidden'
  } else {
    document.body.style.overflow = ''
  }
})

const close = () => {
  emit('update:visible', false)
}

const handleError = (e) => {
  e.target.style.display = 'none'
}
</script>

<style scoped>
.image-viewer-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.image-viewer-container {
  background: white;
  border-radius: 12px;
  max-width: 90vw;
  max-height: 90vh;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: slideUp 0.3s ease;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.image-viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #0066cc;
  color: white;
}

.image-name {
  font-size: 14px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: calc(100% - 40px);
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  font-size: 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s;
  line-height: 1;
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

.image-viewer-wrapper {
  padding: 20px;
  max-height: calc(90vh - 100px);
  overflow: auto;
  background: #f9f9f9;
}

.image-viewer-wrapper img {
  max-width: 100%;
  max-height: calc(90vh - 140px);
  display: block;
  margin: 0 auto;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.image-viewer-footer {
  padding: 10px 20px;
  background: #f5f5f5;
  border-top: 1px solid #e0e0e0;
}

.image-info {
  font-size: 12px;
  color: #666;
}
</style>
