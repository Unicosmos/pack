<template>
  <div class="filter-bar">
    <div class="filter-left">
      <slot name="filters"></slot>
    </div>
    <div class="filter-right">
      <slot name="actions"></slot>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
.filter-bar {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: nowrap;
  justify-content: space-between;
}

.filter-left {
  display: flex;
  gap: 6px;
  align-items: center;
}

.filter-right {
  display: flex;
  align-items: center;
}
</style>
