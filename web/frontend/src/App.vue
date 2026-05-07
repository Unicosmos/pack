<template>
  <div class="app-container">
    <StatusBanner :status="store.systemStatus" />

    <div class="header">
      <h1>📦 Pack Web - 箱货检测与SKU匹配</h1>
      <p>上传图片，自动检测箱体并匹配SKU</p>
    </div>

    <div class="main-content">
      <div class="section">
        <div class="upload-area" :class="{ dragover: isDragover }" @click="triggerUpload" @dragover.prevent="isDragover = true" @dragleave.prevent="isDragover = false" @drop.prevent="handleDrop">
          <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" style="display: none">
          <div class="upload-icon">📤</div>
          <div class="upload-text">点击或拖拽上传图片</div>
          <div class="upload-hint">支持 JPG、PNG 格式</div>
        </div>

        <div v-if="store.selectedFile" class="preview-container show">
          <div class="preview-box">
            <img :src="store.previewUrl" class="preview-img" alt="预览">
            <div class="preview-info">
              <div class="preview-name">{{ store.selectedFile.name }}</div>
              <div class="preview-size">{{ formatFileSize(store.selectedFile.size) }}</div>
            </div>
            <button class="preview-remove" @click="store.removeFile">移除</button>
          </div>
        </div>

        <div class="btn-group">
          <button class="btn btn-success" :disabled="!store.selectedFile || store.isProcessing" @click="processImage">
            {{ store.isProcessing ? '处理中...' : '🔍 开始检测' }}
          </button>
          <button class="btn btn-default" @click="store.reset">🔄 重置</button>
        </div>
      </div>

      <div v-if="store.error" class="section error">
        <div class="error-icon">❌</div>
        <div class="error-text">{{ store.error }}</div>
      </div>

      <div v-if="store.isProcessing" class="section loading">
        <div class="loading-spinner"></div>
        <div>正在处理中，请稍候...</div>
      </div>

      <div v-if="store.result && !store.isProcessing" class="section">
        <div class="result-title">
          <h2>📊 检测结果</h2>
        </div>

        <div class="stats">
          <div class="stat">
            <div class="stat-num">{{ store.result.count || 0 }}</div>
            <div class="stat-label">检测数量</div>
          </div>
          <div class="stat">
            <div class="stat-num" style="color: #67c23a;">{{ store.result.matched_count || 0 }}</div>
            <div class="stat-label">已匹配</div>
          </div>
          <div class="stat">
            <div class="stat-num" style="color: #e6a23c;">{{ store.result.unmatched_count || 0 }}</div>
            <div class="stat-label">未匹配</div>
          </div>
          <div class="stat">
            <div class="stat-num" style="color: #909399;">{{ store.skuCount }}</div>
            <div class="stat-label">SKU库总数</div>
          </div>
        </div>

        <div class="images">
          <div class="image-box">
            <h3 style="margin-bottom: 10px;">检测结果</h3>
            <div class="image-wrapper">
              <img v-if="store.result.image_with_boxes" :src="'data:image/jpeg;base64,' + store.result.image_with_boxes" class="result-image">
              <div v-else class="empty-result">
                <div class="empty-icon">📭</div>
                <p>未检测到目标</p>
              </div>
            </div>
          </div>
        </div>

        <div style="margin-top: 25px;">
          <h3 style="font-size: 16px; font-weight: 600; color: #333; margin-bottom: 15px;">📋 检测详情</h3>
          <div class="detection-list">
            <div v-for="(box, idx) in store.result.boxes" :key="idx" class="detection-item-container">
              <div class="detection-item">
                <img v-if="store.result.crops && store.result.crops[idx]" :src="'data:image/jpeg;base64,' + store.result.crops[idx]" class="thumb" alt="缩略图">
                <div v-else class="thumb" style="background: #f0f0f0; display: flex; align-items: center; justify-content: center;">N/A</div>

                <div class="item-info">
                  <div>
                    <strong>箱体 {{ idx + 1 }}</strong>
                    <span style="margin-left: 10px; color: #666;">置信度: {{ (box.confidence * 100).toFixed(1) }}%</span>
                  </div>
                  <div style="font-size: 12px; color: #999;">位置: [{{ box.bbox.join(', ') }}]</div>
                </div>

                <div v-if="store.result.matches && store.result.matches[idx]">
                  <span :class="store.result.matches[idx].status === 'matched' ? 'tag-success' : 'tag-warning'">
                    {{ store.result.matches[idx].sku_id }} ({{ (store.result.matches[idx].similarity * 100).toFixed(1) }}%)
                  </span>
                </div>
                <div v-else>
                  <span class="tag-info">待匹配</span>
                </div>
              </div>

              <div v-if="store.result.matches && store.result.matches[idx] && store.result.matches[idx].top5_labels && store.result.matches[idx].top5_labels.length > 0" class="top5-section">
                <div class="top5-header">
                  <span class="top5-title">Top-5 匹配候选</span>
                  <span class="top5-status" :class="store.result.matches[idx].status">
                    {{ store.result.matches[idx].status === 'matched' ? '✓ 已匹配' : store.result.matches[idx].status === 'low_conf' ? '⚠️ 低置信' : '✗ 未匹配' }}
                  </span>
                </div>
                <div class="top5-grid">
                  <div
                    v-for="(label, labelIdx) in store.result.matches[idx].top5_labels"
                    :key="labelIdx"
                    class="top5-item"
                    :class="{ 'top1': labelIdx === 0, 'selected': labelIdx === 0 && store.result.matches[idx].status === 'matched' }"
                  >
                    <div class="top5-thumb">
                      <img
                        v-if="label.image_name"
                        :src="`/static/sku_images/${label.sku_id}/${label.image_name}`"
                        :alt="label.sku_name || label.label"
                        onerror="this.style.display='none'"
                      />
                      <div v-if="!label.image_name" class="top5-placeholder">
                        <span class="top5-rank">{{ labelIdx + 1 }}</span>
                      </div>
                    </div>
                    <div class="top5-info">
                      <div class="top5-sku-id">{{ label.sku_id || label.label }}</div>
                      <div class="top5-sku-name">{{ label.sku_name }}</div>
                      <div class="top5-similarity">相似度: {{ (label.similarity * 100).toFixed(1) }}%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="store.isIdle && !store.error" class="section empty">
        <div class="empty-icon">📷</div>
        <p>请上传图片开始检测</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { detectAndMatch } from './api/detector'
import { useAppStore } from './stores/app'
import StatusBanner from './components/StatusBanner.vue'

const store = useAppStore()
const fileInput = ref(null)
const isDragover = ref(false)

const triggerUpload = () => {
  fileInput.value?.click()
}

const handleFileSelect = (e) => {
  const file = e.target.files[0]
  if (file) {
    handleFile(file)
  }
}

const handleDrop = (e) => {
  isDragover.value = false
  const file = e.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    handleFile(file)
  } else {
    ElMessage.warning('请上传图片文件')
  }
}

const handleFile = (file) => {
  store.uploadImage(file)
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

const processImage = async () => {
  if (!store.selectedFile) {
    ElMessage.warning('请先选择图片')
    return
  }

  store.startProcessing()

  try {
    const result = await detectAndMatch(store.selectedFile, 0.5, 0.85)
    store.completeSuccess(result)
    ElMessage.success('检测完成')
  } catch (err) {
    const errorMsg = err.response?.data?.detail || err.message || '处理失败'
    store.completeError(errorMsg)
    ElMessage.error(errorMsg)
  }
}

onMounted(() => {
  store.fetchSystemHealth()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: #f5f5f5;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 25px 30px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header h1 {
  font-size: 24px;
  font-weight: 600;
  margin: 0;
}

.header p {
  font-size: 14px;
  opacity: 0.9;
  margin-top: 5px;
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.section {
  background: white;
  border-radius: 12px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.upload-area {
  border: 2px dashed #667eea;
  border-radius: 10px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-area:hover {
  background: #f0f5ff;
  border-color: #764ba2;
}

.upload-area.dragover {
  background: #e8f0fe;
  border-color: #667eea;
  border-style: solid;
}

.upload-icon {
  font-size: 48px;
  color: #667eea;
  margin-bottom: 15px;
}

.upload-text {
  color: #667eea;
  font-size: 18px;
  margin-bottom: 8px;
}

.upload-hint {
  color: #999;
  font-size: 14px;
}

.preview-container {
  margin-top: 20px;
  display: none;
}

.preview-container.show {
  display: block;
}

.preview-box {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 15px;
  background: #f9f9f9;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.preview-img {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid #ddd;
}

.preview-info {
  flex: 1;
}

.preview-name {
  font-size: 16px;
  color: #333;
  margin-bottom: 5px;
  word-break: break-all;
}

.preview-size {
  font-size: 13px;
  color: #999;
}

.preview-remove {
  padding: 8px 16px;
  background: #ff4d4f;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.preview-remove:hover {
  background: #ff7875;
}

.btn {
  padding: 14px 32px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  margin-right: 10px;
  transition: all 0.3s ease;
}

.btn-success {
  background: linear-gradient(135deg, #67c23a 0%, #52c41a 100%);
  color: white;
}

.btn-success:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(103, 194, 58, 0.4);
}

.btn-default {
  background: #909399;
  color: white;
}

.btn-default:hover {
  background: #7d8085;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-group {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}

.error {
  background: #fef0f0;
  border: 1px solid #fde2e2;
  padding: 20px;
  border-radius: 8px;
  color: #f56c6c;
  display: flex;
  align-items: center;
  gap: 12px;
}

.error-icon {
  font-size: 24px;
}

.error-text {
  font-size: 14px;
}

.loading {
  text-align: center;
  padding: 40px;
  font-size: 18px;
  color: #667eea;
}

.loading-spinner {
  display: inline-block;
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 15px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.stats {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.stat {
  flex: 1;
  min-width: 150px;
  background: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  border: 1px solid #f0f0f0;
}

.stat-num {
  font-size: 32px;
  font-weight: bold;
  color: #667eea;
}

.stat-label {
  color: #666;
  margin-top: 5px;
}

.images {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.image-box {
  flex: 1;
  min-width: 300px;
}

.image-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  background: #f9f9f9;
  border-radius: 8px;
  padding: 10px;
  min-height: 200px;
}

.result-image {
  max-width: 100%;
  max-height: 500px;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.result-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.result-title h2 {
  margin: 0;
}

.detection-list {
  max-height: 400px;
  overflow-y: auto;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.detection-item:hover {
  background: #fafafa;
}

.thumb {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
}

.item-info {
  flex: 1;
}

.tag-success {
  background: #e1f3d8;
  color: #67c23a;
  padding: 4px 12px;
  border-radius: 4px;
  font-weight: 500;
}

.tag-warning {
  background: #faecd8;
  color: #e6a23c;
  padding: 4px 12px;
  border-radius: 4px;
  font-weight: 500;
}

.tag-info {
  background: #e6f1f6;
  color: #909399;
  padding: 4px 12px;
  border-radius: 4px;
}

.detection-item-container {
  margin-bottom: 16px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 12px;
  background: #fafafa;
}

.detection-item:hover {
  background: #f5f5f5;
}

.top5-section {
  padding: 16px;
  background: #fff;
  border-top: 1px solid #f0f0f0;
}

.top5-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.top5-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.top5-status {
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 4px;
}

.top5-status.matched {
  background: #e1f3d8;
  color: #67c23a;
}

.top5-status.low_conf {
  background: #faecd8;
  color: #e6a23c;
}

.top5-status.unmatched {
  background: #fef0f0;
  color: #f56c6c;
}

.top5-grid {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.top5-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px;
  background: #fff;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  width: calc(20% - 8px);
  min-width: 120px;
  transition: all 0.2s ease;
}

.top5-item:hover {
  border-color: #667eea;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
}

.top5-item.top1 {
  border-color: #67c23a;
  background: #f0f9eb;
}

.top5-item.selected {
  border-width: 2px;
  border-color: #67c23a;
}

.top5-thumb {
  width: 80px;
  height: 80px;
  border-radius: 4px;
  overflow: hidden;
  background: #f0f0f0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
}

.top5-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.top5-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e0e0e0;
}

.top5-rank {
  font-size: 24px;
  font-weight: bold;
  color: #999;
}

.top5-info {
  text-align: center;
  width: 100%;
}

.top5-sku-id {
  font-size: 12px;
  font-weight: 600;
  color: #333;
  margin-bottom: 3px;
}

.top5-sku-name {
  font-size: 11px;
  color: #666;
  margin-bottom: 3px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.top5-similarity {
  font-size: 11px;
  color: #999;
}

.empty {
  text-align: center;
  padding: 60px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 15px;
}

.empty-result {
  text-align: center;
  padding: 40px;
  color: #999;
}

.empty-result .empty-icon {
  font-size: 36px;
}
</style>
