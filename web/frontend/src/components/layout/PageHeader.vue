<template>
  <header class="page-header">
    <div class="header-content">
      <h1 class="header-title">{{ title }}</h1>
    </div>
  </header>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true,
    default: ''
  }
})
</script>

<style scoped>
.page-header {
  position: sticky;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  background: var(--color-primary);
  padding: 20px 30px;
  box-shadow: var(--shadow-md);
  transition: all 0.3s ease;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1800px;
  margin: 0 auto;
  width: 100%;
}

.header-title {
  margin: 0;
  font-size: 22px;
  color: white;
  font-weight: 600;
  line-height: 1.2;
}

@media (max-width: 768px) {
  .page-header {
    padding: 14px 16px;
  }

  .header-title {
    font-size: 18px;
  }
}

@media (max-width: 480px) {
  .page-header {
    padding: 12px 16px;
  }

  .header-title {
    font-size: 16px;
  }
}
</style>