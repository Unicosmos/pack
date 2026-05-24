<template>
  <div v-if="imageBase64" class="result-image-wrapper">
    <img :src="'data:image/jpeg;base64,' + imageBase64" class="result-image" @error="handleImageError">
  </div>
</template>

<script setup>
defineProps({
  imageBase64: {
    type: String,
    default: ''
  }
})

const handleImageError = (e) => {
  e.target.style.display = 'none'
}
</script>

<style scoped>
.result-image-wrapper {
  background: #f9f9f9;
  border-radius: 8px;
  padding: 10px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
}

.result-image {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
}
</style>
