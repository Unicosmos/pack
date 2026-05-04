<template>
  <div>
    <div class="header">
      <h1>📦 Pack Web - 箱货检测与SKU匹配</h1>
      <p>上传图片，自动检测箱体并匹配SKU</p>
    </div>

    <div class="main-content">
      <div class="upload-section">
        <el-space wrap>
          <el-upload
            action="#"
            :auto-upload="false"
            :show-file-list="false"
            :on-change="handleFileChange"
            accept="image/*"
          >
            <el-button type="primary" size="large">
              <span>📤 上传图片</span>
            </el-button>
          </el-upload>

          <el-button
            type="success"
            size="large"
            :disabled="!selectedFile || loading"
            @click="processImage"
          >
            {{ loading ? '处理中...' : '🔍 开始检测' }}
          </el-button>

          <el-button @click="reset">
            🔄 重置
          </el-button>
        </el-space>

        <div v-if="selectedFile" style="margin-top: 15px;">
          <el-text>已选择: {{ selectedFile.name }}</el-text>
        </div>
      </div>

      <div class="params-section">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-text>检测置信度阈值: {{ detectConf }}</el-text>
            <el-slider v-model="detectConf" :min="0.1" :max="0.9" :step="0.05" />
          </el-col>
          <el-col :span="12">
            <el-text>匹配相似度阈值: {{ matchThreshold }}</el-text>
            <el-slider v-model="matchThreshold" :min="0.5" :max="0.99" :step="0.05" />
          </el-col>
        </el-row>
      </div>

      <div v-if="loading" style="text-align: center; padding: 40px;">
        <el-icon class="is-loading" style="font-size: 50px; color: #667eea;">
          <Loading />
        </el-icon>
        <p style="margin-top: 15px; color: #666;">处理中，请稍候...</p>
      </div>

      <div v-if="error" style="padding: 20px;">
        <el-alert type="error" :title="error" show-icon />
      </div>

      <div v-if="result && !loading" class="results-section">
        <h2 style="margin-bottom: 20px;">📊 检测结果</h2>

        <el-row :gutter="20" style="margin-bottom: 20px;">
          <el-col :span="6">
            <div class="stats-card">
              <div class="stats-number">{{ result.count || 0 }}</div>
              <div>检测数量</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stats-card">
              <div class="stats-number" style="color: #67c23a;">
                {{ result.matched_count || 0 }}
              </div>
              <div>已匹配</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stats-card">
              <div class="stats-number" style="color: #e6a23c;">
                {{ result.unmatched_count || 0 }}
              </div>
              <div>未匹配</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stats-card">
              <div class="stats-number" style="color: #909399;">
                {{ skuCount }}
              </div>
              <div>SKU库总数</div>
            </div>
          </el-col>
        </el-row>

        <div class="image-container">
          <div class="image-box">
            <h3>原图 + 检测框</h3>
            <img v-if="result.image_with_boxes" :src="'data:image/jpeg;base64,' + result.image_with_boxes" alt="检测结果" />
            <p v-else style="color: #999;">未检测到目标</p>
          </div>
        </div>

        <h3 style="margin-top: 25px; margin-bottom: 15px;">📋 检测详情</h3>
        <div class="detection-list">
          <div v-for="(box, idx) in result.boxes" :key="idx" class="detection-item">
            <img
              v-if="result.crops && result.crops[idx]"
              :src="'data:image/jpeg;base64,' + result.crops[idx]"
              class="crop-thumb"
            />
            <div v-else class="crop-thumb" style="background: #f0f0f0; display: flex; align-items: center; justify-content: center;">
              N/A
            </div>

            <div style="flex: 1;">
              <div>
                <strong>箱体 {{ idx + 1 }}</strong>
                <span style="margin-left: 10px; color: #666;">
                  置信度: {{ (box.confidence * 100).toFixed(1) }}%
                </span>
              </div>
              <div style="font-size: 12px; color: #999;">
                位置: [{{ box.bbox.join(', ') }}]
              </div>
            </div>

            <div v-if="result.matches && result.matches[idx]">
              <el-tag
                :type="getMatchTagType(result.matches[idx].status)"
                size="large"
              >
                {{ result.matches[idx].sku_id }}
                <span style="margin-left: 5px;">
                  ({{ (result.matches[idx].similarity * 100).toFixed(1) }}%)
                </span>
              </el-tag>
            </div>
            <div v-else>
              <el-tag type="info" size="large">待匹配</el-tag>
            </div>
          </div>
        </div>
      </div>

      <div v-if="!result && !loading" style="text-align: center; padding: 60px; color: #999;">
        <p style="font-size: 18px;">👈 请上传图片开始检测</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import axios from 'axios'

const API_BASE = '/api'

const selectedFile = ref(null)
const loading = ref(false)
const error = ref(null)
const result = ref(null)
const skuCount = ref(0)

const detectConf = ref(0.5)
const matchThreshold = ref(0.85)

const getMatchTagType = (status) => {
  switch (status) {
    case 'matched': return 'success'
    case 'unmatched': return 'warning'
    default: return 'info'
  }
}

const handleFileChange = (file) => {
  selectedFile.value = file.raw
  error.value = null
  result.value = null
}

const reset = () => {
  selectedFile.value = null
  error.value = null
  result.value = null
}

const processImage = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择图片')
    return
  }

  loading.value = true
  error.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('conf_threshold', detectConf.value)
    formData.append('match_threshold', matchThreshold.value)

    const response = await axios.post(`${API_BASE}/detect-and-match`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })

    result.value = response.data
    await fetchSkuCount()
    ElMessage.success('检测完成')
  } catch (err) {
    console.error(err)
    error.value = err.response?.data?.detail || err.message || '处理失败'
    ElMessage.error(error.value)
  } finally {
    loading.value = false
  }
}

const fetchSkuCount = async () => {
  try {
    const res = await axios.get(`${API_BASE}/skus`)
    skuCount.value = res.data.count || 0
  } catch {
    skuCount.value = 0
  }
}
</script>
