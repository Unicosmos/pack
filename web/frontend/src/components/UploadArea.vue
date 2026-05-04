<template>
  <div class="upload-area">
    <el-upload
      action="#"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFileChange"
      :drag="drag"
      accept="image/*"
    >
      <div v-if="drag" class="upload-drag">
        <el-icon class="upload-icon"><Upload /></el-icon>
        <div class="upload-text">将图片拖到此处，或<em>点击上传</em></div>
        <div class="upload-hint">支持 JPG、PNG 格式</div>
      </div>
      <slot v-else>
        <el-button type="primary" size="large">
          <el-icon style="margin-right: 8px;"><Upload /></el-icon>
          选择图片
        </el-button>
      </slot>
    </el-upload>
  </div>
</template>

<script setup>
import { Upload } from '@element-plus/icons-vue'

const props = defineProps({
  drag: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['change'])

const handleFileChange = (file) => {
  emit('change', file.raw)
}
</script>

<style scoped>
.upload-area {
  display: inline-block;
}

.upload-drag {
  padding: 40px;
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  background: #fafafa;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.3s;
}

.upload-drag:hover {
  border-color: #667eea;
}

.upload-icon {
  font-size: 48px;
  color: #999;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 14px;
  color: #666;
}

.upload-text em {
  color: #667eea;
  font-style: normal;
}

.upload-hint {
  font-size: 12px;
  color: #999;
  margin-top: 8px;
}
</style>