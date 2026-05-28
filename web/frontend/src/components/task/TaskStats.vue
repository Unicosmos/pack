<template>
  <div class="task-stats">
    <div class="section">
      <div class="chart-section">
        <div class="chart-header">
          <h3>📋 任务状态分布</h3>
        </div>
        <div class="stacked-bar-container">
          <div class="stacked-bar">
            <div 
              class="bar-segment completed" 
              :style="{ width: completedPercent + '%' }"
            ></div>
            <div 
              class="bar-segment pending" 
              :style="{ width: pendingPercent + '%' }"
            ></div>
            <div 
              class="bar-segment detected" 
              :style="{ width: detectedPercent + '%' }"
            ></div>
            <div 
              class="bar-segment failed" 
              :style="{ width: failedPercent + '%' }"
            ></div>
          </div>
          <div class="bar-legend">
            <span class="legend-item">
              <span class="legend-dot completed"></span>
              <span>已完成 {{ stats.completed || 0 }}</span>
            </span>
            <span class="legend-item">
              <span class="legend-dot pending"></span>
              <span>待识别 {{ stats.pending || 0 }}</span>
            </span>
            <span class="legend-item">
              <span class="legend-dot detected"></span>
              <span>待审核 {{ stats.detected || 0 }}</span>
            </span>
            <span class="legend-item">
              <span class="legend-dot failed"></span>
              <span>识别失败 {{ stats.failed || 0 }}</span>
            </span>
            <span class="legend-item total">
              <span>总任务: {{ stats.total || 0 }}</span>
            </span>
          </div>
        </div>
      </div>

      <div class="chart-section mt-4">
        <div class="chart-header">
          <h3>📦 箱体匹配进度</h3>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div 
              class="progress-fill matched" 
              :style="{ width: boxMatchPercent + '%' }"
            ></div>
            <div 
              class="progress-fill unmatched" 
              :style="{ width: (100 - boxMatchPercent) + '%' }"
            ></div>
            <span class="progress-rate">{{ stats.warehouse?.match_rate || 0 }}%</span>
          </div>
          <div class="progress-legend">
            <span class="legend-item">
              <span class="legend-dot matched"></span>
              <span>已匹配 {{ stats.warehouse?.matched_boxes || 0 }}</span>
            </span>
            <span class="legend-item">
              <span class="legend-dot unmatched"></span>
              <span>未匹配 {{ stats.warehouse?.unmatched_boxes || 0 }}</span>
            </span>
            <span class="legend-item total">
              <span>箱体总数: {{ stats.warehouse?.total_boxes || 0 }}</span>
            </span>
          </div>
        </div>
      </div>

      <div v-if="stats.sku?.distribution?.length > 0" class="chart-section mt-4">
        <div class="chart-header collapsible" @click="skuCollapsed = !skuCollapsed">
          <h3>📊 识别结果汇总（商品SKU及数量）</h3>
          <div class="chart-header-right">
            <span class="chart-subtitle">共 {{ stats.sku?.category_count || 0 }} 种商品</span>
            <span class="collapse-icon">{{ skuCollapsed ? '▶' : '▼' }}</span>
          </div>
        </div>
        <div v-show="!skuCollapsed" class="horizontal-bars">
          <div 
            v-for="(item, index) in stats.sku.distribution" 
            :key="item.sku_id" 
            class="horizontal-bar-item"
          >
            <div class="bar-label">
              <span class="rank">{{ index + 1 }}</span>
              <span class="sku-name">{{ item.sku_name || '未知' }}</span>
              <span class="sku-id">({{ item.sku_id }})</span>
            </div>
            <div class="bar-track">
              <div 
                class="bar-fill" 
                :style="{ width: getSkuBarWidth(item.count) + '%' }"
              ></div>
            </div>
            <span class="bar-value">{{ item.count }}个</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  stats: {
    type: Object,
    required: true
  }
})

const skuCollapsed = ref(false)

const completedPercent = computed(() => {
  const total = props.stats.total || 0
  const completed = props.stats.completed || 0
  return total > 0 ? (completed / total) * 100 : 0
})

const detectedPercent = computed(() => {
  const total = props.stats.total || 0
  const detected = props.stats.detected || 0
  return total > 0 ? (detected / total) * 100 : 0
})

const pendingPercent = computed(() => {
  const total = props.stats.total || 0
  const pending = props.stats.pending || 0
  return total > 0 ? (pending / total) * 100 : 0
})

const failedPercent = computed(() => {
  const total = props.stats.total || 0
  const failed = props.stats.failed || 0
  return total > 0 ? (failed / total) * 100 : 0
})

const boxMatchPercent = computed(() => {
  return props.stats.warehouse?.match_rate || 0
})

const maxSkuCount = computed(() => {
  const distribution = props.stats.sku?.distribution || []
  return Math.max(...distribution.map(item => item.count), 1)
})

const getSkuBarWidth = (count) => {
  return (count / maxSkuCount.value) * 100
}
</script>

<style scoped>
.task-stats {
  width: 100%;
}

.section {
  background: var(--color-bg-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  box-shadow: var(--shadow-md);
}

.mt-4 {
  margin-top: 16px;
}

.chart-section {
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-md);
  padding: var(--spacing-lg);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.chart-header.collapsible {
  cursor: pointer;
  user-select: none;
}

.chart-header-right {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.collapse-icon {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  transition: transform var(--transition-fast);
}

.chart-header h3 {
  margin: 0;
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
}

.chart-subtitle {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.stacked-bar-container {
  width: 100%;
}

.stacked-bar {
  display: flex;
  height: 24px;
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.bar-segment {
  height: 100%;
  transition: width var(--transition-normal);
}

.bar-segment.completed {
  background: linear-gradient(90deg, #67c23a, #85ce61);
}

.bar-segment.pending {
  background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}

.bar-segment.detected {
  background: linear-gradient(90deg, #e6a23c, #f0c78a);
}

.bar-segment.failed {
  background: linear-gradient(90deg, #f56c6c, #f89898);
}

.bar-legend {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--spacing-sm);
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.legend-item.total {
  font-weight: 600;
  color: var(--color-text-primary);
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-dot.completed {
  background: #67c23a;
}

.legend-dot.detected {
  background: #e6a23c;
}

.legend-dot.pending {
  background: #0ea5e9;
}

.legend-dot.failed {
  background: #f56c6c;
}

.legend-dot.matched {
  background: #67c23a;
}

.legend-dot.unmatched {
  background: #909399;
}

.progress-bar-container {
  width: 100%;
}

.progress-bar {
  position: relative;
  height: 32px;
  background: #e4e7ed;
  border-radius: var(--radius-sm);
  overflow: hidden;
  display: flex;
}

.progress-fill {
  height: 100%;
  transition: width var(--transition-normal);
}

.progress-fill.matched {
  background: linear-gradient(90deg, #67c23a, #85ce61);
}

.progress-fill.unmatched {
  background: #d9d9d9;
}

.progress-rate {
  position: absolute;
  right: var(--spacing-sm);
  top: 50%;
  transform: translateY(-50%);
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--color-text-primary);
  text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.progress-legend {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--spacing-sm);
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.horizontal-bars {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.horizontal-bar-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-sm);
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  transition: background-color var(--transition-fast);
}

.horizontal-bar-item:hover {
  background: var(--color-bg-primary);
}

.bar-label {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  width: 200px;
  flex-shrink: 0;
}

.bar-label .rank {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-primary);
  color: white;
  border-radius: 50%;
  font-size: var(--font-size-xs);
  font-weight: bold;
  margin-right: var(--spacing-md);
}

.bar-label .sku-name {
  font-size: var(--font-size-sm);
  color: var(--color-text-primary);
  font-weight: 500;
}

.bar-label .sku-id {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.bar-track {
  flex: 1;
  height: 16px;
  background: #e4e7ed;
  border-radius: var(--radius-xs);
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
  border-radius: var(--radius-xs);
  transition: width var(--transition-normal);
}

.bar-value {
  width: 60px;
  text-align: right;
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--color-text-primary);
}
</style>