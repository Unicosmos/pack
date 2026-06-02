<template>
  <div class="task-stats">
    <!-- 可折叠统计面板 -->
    <div class="stats-panel">
      <div class="stats-toggle" @click="toggleStats">
        <div class="stats-toggle-left">
          <span>📊</span>
          <span>识别统计概览</span>
          <span class="stats-toggle-icon" :class="{ expanded: !collapsed }">
            {{ collapsed ? '▼' : '▲' }}
          </span>
        </div>
        <div class="stats-toggle-right">
          已完成 {{ stats.completed || 0 }} / 待识别 {{ stats.pending || 0 }} / 待审核 {{ stats.detected || 0 }}
        </div>
      </div>

      <div class="stats-content" :class="{ expanded: !collapsed }">
        <div class="stats-inner">
          <!-- 任务状态分布 -->
          <div class="stats-section">
            <div class="stats-section-title">任务状态分布</div>
            <div class="status-distribution">
              <div class="status-chip done">
                <span class="num">{{ stats.completed || 0 }}</span>
                <span class="label">已完成</span>
              </div>
              <div class="status-chip pending">
                <span class="num">{{ stats.pending || 0 }}</span>
                <span class="label">待识别</span>
              </div>
              <div class="status-chip review">
                <span class="num">{{ stats.detected || 0 }}</span>
                <span class="label">待审核</span>
              </div>
              <div class="status-chip fail">
                <span class="num">{{ stats.failed || 0 }}</span>
                <span class="label">失败</span>
              </div>
            </div>
          </div>

          <!-- 箱体匹配进度 -->
          <div class="stats-section">
            <div class="stats-section-title">箱体匹配进度</div>
            <div class="box-progress">
              <div class="box-progress-header">
                <span>已匹配 {{ stats.warehouse?.matched_boxes || 0 }}</span>
                <span>未匹配 {{ stats.warehouse?.unmatched_boxes || 0 }}</span>
                <span>总数 {{ stats.warehouse?.total_boxes || 0 }}</span>
              </div>
              <div class="progress-bar-bg">
                <div
                  class="progress-bar-fill"
                  :style="{ width: boxMatchPercent + '%' }"
                ></div>
              </div>
            </div>
          </div>

          <!-- SKU识别排名 -->
          <div v-if="stats.sku?.distribution?.length > 0" class="stats-section">
            <div class="stats-section-title">🏆 SKU识别排名</div>
            <div class="sku-ranking">
              <div
                v-for="(item, index) in stats.sku.distribution.slice(0, 8)"
                :key="item.sku_id"
                class="sku-item"
              >
                <div class="sku-rank">{{ index + 1 }}</div>
                <div class="sku-info">
                  <div class="sku-name">{{ item.sku_name || '未知' }}</div>
                  <div class="sku-id">({{ item.sku_id }})</div>
                </div>
                <div class="sku-bar-wrap">
                  <div class="sku-bar-bg">
                    <div
                      class="sku-bar-fill"
                      :style="{ width: getSkuBarWidth(item.count) + '%' }"
                    ></div>
                  </div>
                </div>
                <div class="sku-count">{{ item.count }}</div>
              </div>
            </div>
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

const collapsed = ref(true)

const toggleStats = () => {
  collapsed.value = !collapsed.value
}

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

/* 可折叠面板 */
.stats-panel {
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.stats-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  cursor: pointer;
  background: var(--color-bg-tertiary);
  transition: all 0.2s;
  user-select: none;
}

.stats-toggle:hover {
  background: var(--color-bg-hover);
}

.dark .stats-toggle {
  background: rgba(0, 0, 0, 0.1);
}

.dark .stats-toggle:hover {
  background: rgba(0, 0, 0, 0.2);
}

.stats-toggle-left {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.stats-toggle-icon {
  font-size: 11px;
  color: var(--color-text-tertiary);
  transition: transform 0.3s;
}

.stats-toggle-icon.expanded {
  transform: rotate(180deg);
}

.stats-toggle-right {
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.stats-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.stats-content.expanded {
  max-height: 600px;
}

.stats-inner {
  padding: 0 16px 16px;
}

/* 统计区块 */
.stats-section {
  margin-top: 16px;
}

.stats-section:first-child {
  margin-top: 0;
}

.stats-section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* 状态分布卡片 */
.status-distribution {
  display: flex;
  gap: 6px;
}

.status-chip {
  flex: 1;
  padding: 8px 4px;
  border-radius: 6px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  text-align: center;
}

.status-chip .num {
  font-size: 18px;
  font-weight: 700;
  display: block;
  margin-bottom: 1px;
}

.status-chip .label {
  font-size: 10px;
  color: var(--color-text-tertiary);
}

.status-chip.done .num {
  color: var(--color-success);
}

.status-chip.pending .num {
  color: var(--color-primary);
}

.status-chip.review .num {
  color: var(--color-warning);
}

.status-chip.fail .num {
  color: var(--color-text-tertiary);
}

/* 箱体匹配进度 */
.box-progress {
  margin-bottom: 4px;
}

.box-progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.progress-bar-bg {
  height: 6px;
  background: var(--color-bg-secondary);
  border-radius: 3px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-success), #4ade80);
  border-radius: 3px;
  transition: width 0.5s ease;
}

/* SKU排名 */
.sku-ranking {
  max-height: 200px;
  overflow-y: auto;
}

.sku-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--color-border-light);
}

.dark .sku-item {
  border-bottom-color: rgba(55, 65, 81, 0.5);
}

.sku-item:last-child {
  border-bottom: none;
}

.sku-rank {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--color-primary);
  color: #fff;
  font-size: 10px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.sku-info {
  flex: 1;
  min-width: 0;
}

.sku-name {
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--color-text-primary);
}

.sku-id {
  font-size: 10px;
  color: var(--color-text-tertiary);
}

.sku-bar-wrap {
  width: 80px;
  flex-shrink: 0;
}

.sku-bar-bg {
  height: 5px;
  background: var(--color-bg-secondary);
  border-radius: 3px;
  overflow: hidden;
}

.sku-bar-fill {
  height: 100%;
  background: var(--color-primary);
  border-radius: 3px;
  transition: width 0.5s ease;
}

.sku-count {
  width: 32px;
  text-align: right;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  flex-shrink: 0;
}

/* 深色模式适配 */
.dark .status-chip {
  background: rgba(0, 0, 0, 0.2);
}

.dark .progress-bar-bg {
  background: rgba(0, 0, 0, 0.3);
}

.dark .sku-bar-bg {
  background: rgba(0, 0, 0, 0.3);
}
</style>
