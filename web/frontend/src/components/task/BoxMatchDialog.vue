<template>
  <Teleport to="body">
    <div v-if="visible" class="dialog-overlay" @click.self="$emit('close')">
      <div class="dialog-container">
        <div class="dialog-header">
          <h3>箱体 {{ boxIndex + 1 }} - 修改匹配</h3>
          <button class="dialog-close" @click="$emit('close')">×</button>
        </div>

        <div class="dialog-body">
          <div class="box-preview">
            <SkuImage
              :image-path="getBoxImageUrl(box)"
              height="120px"
              fit="contain"
            />
          </div>

          <div class="current-match">
            <span class="current-label">当前匹配：</span>
            <span v-if="currentMatch" :class="['current-value', currentMatch.status]">
              {{ currentMatch.sku_id || '未匹配' }}
            </span>
            <span v-else class="current-value unmatched">未匹配</span>
          </div>

          <div v-if="top5Labels.length > 0" class="top5-section">
            <div class="top5-title">Top-5 候选</div>
            <div class="top5-list">
              <div
                v-for="(label, lIdx) in top5Labels"
                :key="lIdx"
                class="top5-item"
                :class="{ 'selected': selectedSku === (label.sku_id || label.label), 'top1': lIdx === 0 }"
                @click="selectSku(label)"
              >
                <div class="top5-rank">{{ lIdx + 1 }}</div>
                <SkuImage
                  :image-path="getImageUrlFromPath(label.image_path)"
                  width="40px"
                  height="40px"
                  fit="cover"
                  class="top5-thumb"
                />
                <div class="top5-info">
                  <div class="top5-sku">{{ label.sku_id || label.label }}</div>
                  <div class="top5-name">{{ label.name || label.sku_name || '' }}</div>
                </div>
                <div class="top5-sim">{{ (label.similarity * 100).toFixed(1) }}%</div>
              </div>
            </div>
          </div>

          <div class="manual-section">
            <div class="manual-title">或手动输入SKU</div>
            <input
              v-model="manualSku"
              type="text"
              class="manual-input"
              placeholder="输入SKU编号..."
              @input="onManualInput"
            />
          </div>
        </div>

        <div class="dialog-footer">
          <button class="btn btn-default" @click="$emit('close')">取消</button>
          <button class="btn btn-primary" :disabled="!selectedSku && !manualSku" @click="confirmMatch">确认</button>
          <button class="btn btn-submit" @click="submitReview">提交审核</button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import SkuImage from '@sku/SkuImage.vue'
import { getImageUrlFromPath } from '@api/client'

const props = defineProps({
  visible: { type: Boolean, default: false },
  box: { type: Object, default: null },
  boxIndex: { type: Number, default: 0 },
  taskId: { type: [Number, String], default: null }
})

const emit = defineEmits(['close', 'update', 'submit-review'])

const selectedSku = ref(null)
const manualSku = ref('')

const currentMatch = computed(() => {
  return props.box?.match_result || null
})

const top5Labels = computed(() => {
  return currentMatch.value?.top5_labels?.slice(0, 5) || []
})

const getBoxImageUrl = (box) => {
  if (!box) return ''
  if (box.crop_base64) {
    return { url: 'data:image/jpeg;base64,' + box.crop_base64 }
  }
  if (box.crop_path) {
    return getImageUrlFromPath(box.crop_path)
  }
  return ''
}

watch(() => props.visible, (val) => {
  if (val) {
    selectedSku.value = currentMatch.value?.sku_id || null
    manualSku.value = ''
  }
})

const selectSku = (label) => {
  selectedSku.value = label.sku_id || label.label
  manualSku.value = label.sku_id || label.label
}

const onManualInput = () => {
  selectedSku.value = manualSku.value || null
}

const confirmMatch = () => {
  const sku = manualSku.value?.trim() || selectedSku.value
  if (!sku) {
    ElMessage.warning('请选择或输入SKU')
    return
  }

  emit('update', {
    boxId: props.box?.box_id,
    skuId: sku,
    boxIndex: props.boxIndex
  })
  emit('close')
}

const submitReview = () => {
  const sku = manualSku.value?.trim() || selectedSku.value
  if (sku) {
    emit('update', {
      boxId: props.box?.box_id,
      skuId: sku,
      boxIndex: props.boxIndex
    })
  }
  emit('submit-review')
  emit('close')
}
</script>

<style scoped>
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
}

.dialog-container {
  background: var(--color-bg-primary);
  border-radius: 12px;
  width: 480px;
  max-width: 90vw;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow-xl);
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.dialog-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-primary);
}

.dialog-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--color-text-secondary);
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color var(--transition-fast);
}

.dialog-close:hover {
  background: var(--color-bg-tertiary);
}

.dialog-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.box-preview {
  width: 100%;
  background: var(--color-bg-tertiary);
  border-radius: 8px;
  overflow: hidden;
}

.current-match {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.current-label {
  color: var(--color-text-secondary);
}

.current-value {
  font-weight: 600;
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-size: 14px;
}

.current-value.matched {
  background: rgba(103, 194, 58, 0.1);
  color: var(--color-success);
}

.current-value.unmatched {
  background: rgba(245, 108, 108, 0.1);
  color: var(--color-danger);
}

.current-value.low_conf {
  background: rgba(230, 162, 60, 0.1);
  color: var(--color-warning);
}

.top5-section {
  background: var(--color-bg-tertiary);
  border-radius: 8px;
  padding: 12px;
}

.top5-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 10px;
}

.top5-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.top5-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.top5-item:hover {
  border-color: var(--color-primary);
  transform: translateY(-1px);
}

.top5-item.selected {
  border-color: var(--color-primary);
  background: rgba(102, 126, 234, 0.08);
}

.top5-item.top1 {
  border-color: var(--color-primary);
  background: rgba(102, 126, 234, 0.05);
}

.top5-rank {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg-secondary);
  border-radius: 50%;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  flex-shrink: 0;
}

.top5-item.top1 .top5-rank {
  background: var(--color-primary);
  color: white;
}

.top5-thumb {
  width: 40px;
  flex-shrink: 0;
  border-radius: 4px;
  overflow: hidden;
}

.top5-info {
  flex: 1;
  min-width: 0;
}

.top5-sku {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.top5-name {
  font-size: 11px;
  color: var(--color-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.top5-sim {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-success);
  flex-shrink: 0;
}

.manual-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.manual-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.manual-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  outline: none;
  transition: border-color var(--transition-fast);
}

.manual-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}

.btn {
  padding: 8px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-primary {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.btn-default {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
}

.btn-default:hover {
  background: var(--color-bg-secondary);
}

.btn-submit {
  background: linear-gradient(135deg, var(--color-success, #67c23a) 0%, #85ce61 100%);
  color: white;
}

.btn-submit:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(103, 194, 58, 0.3);
}
</style>
