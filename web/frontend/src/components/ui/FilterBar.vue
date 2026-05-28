<template>
  <div class="filter-bar">
    <slot></slot>
  </div>
</template>

<script setup>
</script>

<style scoped>
.filter-bar {
  display: flex;
  gap: var(--spacing-md);
  align-items: center;
  flex-wrap: wrap;
}
</style>
