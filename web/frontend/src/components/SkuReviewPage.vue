<template>
  <div class="sku-review-page">
    <div class="header">
      <h1>📦 SKU 人工审核</h1>
      <button class="btn-primary" @click="saveDatabase">💾 保存更新</button>
    </div>

    <div class="main-content">
      <!-- 左侧：文件夹导航栏 -->
      <div class="folder-nav">
        <div class="folder-nav-header">
          <h3>� 文件夹</h3>
          <div class="folder-nav-info">
            {{ currentFolderIndex + 1 }} / {{ folders.length }}
          </div>
        </div>
        <div class="folder-list">
          <div
            v-for="(folder, idx) in folders"
            :key="folder"
            :class="['folder-item', { active: currentFolderIndex === idx }]"
            @click="selectFolder(idx)"
          >
            <span class="folder-icon">📁</span>
            <span class="folder-name">{{ folder }}</span>
          </div>
        </div>
      </div>

      <!-- 右侧：待审核图片区域 -->
      <div class="left-panel">
        <div class="panel-header">
          <h2>📋 待审核图片 - {{ currentFolder }}</h2>
          <div class="crop-count">共 {{ currentImages.length }} 张图片</div>
        </div>
        <div class="crop-gallery">
          <div
            v-for="(img, idx) in currentImages"
            :key="idx"
            :class="['crop-item', { selected: selectedCrops.includes(idx) }]"
            @click="toggleCropSelect(idx)"
          >
            <img :src="encodeURI(img.url)" :alt="img.name" loading="lazy" @error="(e) => e.target.style.display = 'none'" />
            <div class="crop-name">{{ img.name }}</div>
          </div>
        </div>

        <div class="selected-crops">
          <div class="selected-header">
            <h3>✅ 已选择图片 ({{ selectedCrops.length }})</h3>
            <button class="btn btn-sm btn-secondary" @click="clearSelection">🧹 清空</button>
          </div>
          <div class="selected-gallery">
            <div
              v-for="(idx, i) in selectedCrops"
              :key="i"
              class="selected-item"
              @click="deselectCrop(i)"
            >
              <img 
                :src="encodeURI(currentImages[idx]?.url)" 
                :alt="currentImages[idx]?.name" 
                @error="(e) => e.target.style.display = 'none'"
                @click.stop="openLargeImage(currentImages[idx]?.url, currentImages[idx]?.name, $event)"
              />
              <div class="remove-badge">×</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧：SKU 库区域 -->
      <div class="right-panel">
        <div class="panel-header">
          <h2>🗂️ SKU 库</h2>
          <div class="sku-controls">
            <div class="search-box">
              <input v-model="searchKeyword" placeholder="搜索 SKU 编号或名称" @input="debounceSearch" />
            </div>
            <div class="new-sku-box">
              <input v-model="newSkuInput" placeholder="新增 SKU" />
              <button class="btn btn-success" @click="createSku">➕ 新增</button>
            </div>
          </div>
        </div>
        <div class="sku-gallery">
          <div
            v-for="(sku, idx) in filteredSkus"
            :key="sku.id"
            :class="['sku-item', { selected: selectedSkuId === sku.id }]"
            @click="selectSku(sku)"
          >
            <div class="sku-cover">
              <img v-if="sku.cover_url" :src="encodeURI(sku.cover_url)" :alt="sku.name" @error="(e) => e.target.style.display = 'none'" />
              <div v-else class="no-cover">📷</div>
            </div>
            <div class="sku-info">
              <div class="sku-id">{{ sku.id }}</div>
              <div class="sku-name">{{ sku.name }}</div>
              <div class="sku-count">{{ sku.cnt }}张</div>
            </div>
          </div>
        </div>

        <div class="action-hint">{{ actionHint }}</div>
        <div class="action-buttons">
          <button class="btn btn-success" @click="assignImages" :disabled="!canAssign">✅ 确认归类</button>
          <button class="btn btn-secondary" disabled title="功能已禁用">↩️ 撤回</button>
          <button class="btn btn-danger" disabled title="功能已禁用">🗑️ 删除</button>
        </div>

        <!-- SKU 详情 -->
        <div v-if="selectedSkuId" class="sku-detail">
          <div class="detail-header">
            <h3>SKU 详情：{{ selectedSkuId }}</h3>
            <div v-if="selectedSku" class="sku-info-detail">
              <span class="sku-name-detail">{{ selectedSku.name }}</span>
              <span class="sku-count-detail">{{ getSkuImages().length }} 张图片</span>
            </div>
          </div>
          <div class="sku-detail-images">
            <div
              v-for="(img, idx) in getSkuImages()"
              :key="idx"
              :class="['detail-image', { selected: selectedSkuImages.includes(idx) }]"
              @click="toggleSkuImageSelect(idx)"
            >
              <img 
                :src="encodeURI(img.url)" 
                :alt="img.name" 
                @error="(e) => e.target.style.display = 'none'"
                @click.stop="openLargeImage(img.url, img.name, $event)"
              />
              <div class="image-name">{{ img.name }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="log-panel">
      <div class="panel-header">
        <h3>📜 操作日志</h3>
      </div>
      <div class="logs">
        <div v-for="(log, idx) in logs.slice(-50).reverse()" :key="idx" class="log-item">
          {{ log }}
        </div>
      </div>
    </div>

    <div v-if="showToast" :class="['toast', toastType]">
      {{ toastMessage }}
    </div>

    <!-- 大图查看模态框 -->
    <div v-if="showLargeImage" class="large-image-overlay" @click="closeLargeImage">
      <div class="large-image-container" @click.stop>
        <div class="large-image-header">
          <span class="large-image-name">{{ largeImageName }}</span>
          <button class="close-btn" @click="closeLargeImage">×</button>
        </div>
        <div class="large-image-wrapper">
          <img :src="largeImageUrl" :alt="largeImageName" @error="(e) => e.target.style.display = 'none'" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue';
import { skuReview, getImageUrl, getImageUrlFromPath } from '../api/client';

// 数据状态
const folders = ref([]);
const currentFolderIndex = ref(0);
const currentFolder = ref('');
const currentImages = ref([]);
const selectedCrops = ref([]);
const skus = ref([]);
const selectedSkuId = ref('');
const selectedSku = ref(null);
const selectedSkuImages = ref([]);
const skuImagesData = ref([]);
const searchKeyword = ref('');
const newSkuInput = ref('');
const logs = ref([]);
const showToast = ref(false);
const toastMessage = ref('');
const toastType = ref('info');

// 大图查看
const showLargeImage = ref(false);
const largeImageUrl = ref('');
const largeImageName = ref('');

// 计算属性
const filteredSkus = computed(() => {
  if (!searchKeyword.value) return skus.value;
  const kw = searchKeyword.value.toLowerCase();
  return skus.value.filter(s =>
    s.id.toLowerCase().includes(kw) ||
    s.name.toLowerCase().includes(kw)
  );
});

const actionHint = computed(() => {
  if (!selectedSkuId.value) return '操作提示：先在左侧选择图片，再点击右侧目标SKU';
  const sku = skus.value.find(s => s.id === selectedSkuId.value);
  if (selectedCrops.value.length > 0) {
    return `即将把 ${selectedCrops.value.length} 张图片归类至【${selectedSkuId.value}】(${sku?.name})`;
  }
  return `已选中 SKU: ${selectedSkuId.value}（${sku?.cnt}张）`;
});

const canAssign = computed(() => selectedCrops.value.length > 0 && selectedSkuId.value);
const canRecall = computed(() => selectedSkuImages.value.length > 0);

// Toast 提示
const showToastMsg = (msg, type = 'info') => {
  toastMessage.value = msg;
  toastType.value = type;
  showToast.value = true;
  setTimeout(() => showToast.value = false, 3000);
};

// 添加日志
const addLog = (msg) => {
  const time = new Date().toLocaleTimeString();
  logs.value.push(`[${time}] ${msg}`);
};

// 加载文件夹
const loadFolders = async () => {
  try {
    const res = await skuReview.getFolders();
    if (res.success) {
      folders.value = res.folders;
      if (folders.value.length > 0 && !currentFolder.value) {
        currentFolder.value = folders.value[0];
        currentFolderIndex.value = 0;
        await loadFolderImages();
      }
    }
  } catch (e) {
    addLog('⚠️ 加载文件夹失败');
  }
};

const loadFolderImages = async () => {
  if (!currentFolder.value) return;
  try {
    const res = await skuReview.getFolderImages(currentFolder.value);
    if (res.success) {
      currentImages.value = res.images;
      selectedCrops.value = [];
      addLog(`切换文件夹 → ${currentFolder.value}（${currentImages.value.length}张图片）`);
    }
  } catch (e) {
    addLog('⚠️ 加载图片失败');
  }
};

const prevFolder = async () => {
  if (currentFolderIndex.value > 0) {
    currentFolderIndex.value--;
    currentFolder.value = folders.value[currentFolderIndex.value];
    await loadFolderImages();
  }
};

const nextFolder = async () => {
  if (currentFolderIndex.value < folders.value.length - 1) {
    currentFolderIndex.value++;
    currentFolder.value = folders.value[currentFolderIndex.value];
    await loadFolderImages();
  }
};

const selectFolder = async (idx) => {
  if (currentFolderIndex.value !== idx) {
    currentFolderIndex.value = idx;
    currentFolder.value = folders.value[idx];
    await loadFolderImages();
  }
};

const onFolderSelect = async () => {
  const idx = folders.value.indexOf(currentFolder.value);
  if (idx >= 0) {
    currentFolderIndex.value = idx;
    await loadFolderImages();
  }
};

// 加载SKU
const loadSkus = async () => {
  try {
    const res = await skuReview.getSkus(searchKeyword.value);
    if (res.success) {
      skus.value = res.skus;
    }
  } catch (e) {
    addLog('⚠️ 加载 SKU 列表失败');
  }
};

// 防抖搜索
let searchTimer = null;
const debounceSearch = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadSkus, 300);
};

// 选择操作
const toggleCropSelect = (idx) => {
  const i = selectedCrops.value.indexOf(idx);
  if (i >= 0) {
    selectedCrops.value.splice(i, 1);
  } else {
    selectedCrops.value.push(idx);
  }
  selectedCrops.value.sort((a, b) => a - b);
};

const deselectCrop = (pos) => {
  const idx = selectedCrops.value[pos];
  const i = selectedCrops.value.indexOf(idx);
  if (i >= 0) {
    selectedCrops.value.splice(i, 1);
  }
};

const clearSelection = () => {
  selectedCrops.value = [];
  addLog('🧹 已清空选择');
};

const selectSku = async (sku) => {
  selectedSkuId.value = sku.id;
  selectedSku.value = sku;
  selectedSkuImages.value = [];
  await loadSkuImages(sku.id);
  addLog(`选中 SKU: ${sku.id}（${sku.cnt}张图片）`);
};

const loadSkuImages = async (skuId) => {
  try {
    const res = await skuReview.getSkuImages(skuId);
    if (res.success) {
      skuImagesData.value = res.images;
      selectedSkuImages.value = [];
    }
  } catch (e) {
    addLog('⚠️ 加载 SKU 图片失败');
  }
};

const getSkuImages = () => skuImagesData.value || [];

const toggleSkuImageSelect = (idx) => {
  const i = selectedSkuImages.value.indexOf(idx);
  if (i >= 0) {
    selectedSkuImages.value.splice(i, 1);
  } else {
    selectedSkuImages.value.push(idx);
  }
};

// 操作
const assignImages = async () => {
  try {
    const imagePaths = selectedCrops.value.map(idx => currentImages.value[idx].path);
    const res = await skuReview.assignImages(selectedSkuId.value, imagePaths);
    if (res.success) {
      addLog(`✅ 归类 ${selectedCrops.value.length} 张图片至 ${selectedSkuId.value}`);
      showToastMsg(res.message, 'success');
      selectedCrops.value = [];
      await loadSkus();
      await loadSkuImages(selectedSkuId.value);
    } else {
      showToastMsg(res.detail || '操作失败', 'error');
    }
  } catch (e) {
    showToastMsg('操作失败', 'error');
  }
};

const recallImages = async () => {
  try {
    const images = getSkuImages();
    const imagePaths = selectedSkuImages.value.map(idx => images[idx].path);
    const res = await skuReview.recallImages(selectedSkuId.value, imagePaths);
    if (res.success) {
      addLog(`↩️ 从 ${selectedSkuId.value} 撤回 ${selectedSkuImages.value.length} 张图片`);
      showToastMsg(res.message, 'success');
      selectedSkuImages.value = [];
      await loadSkus();
      await loadSkuImages(selectedSkuId.value);
    } else {
      showToastMsg(res.detail || '操作失败', 'error');
    }
  } catch (e) {
    showToastMsg('操作失败', 'error');
  }
};

const createSku = async () => {
  try {
    let res;
    if (newSkuInput.value.includes('|')) {
      const [oldId, newName] = newSkuInput.value.split('|');
      res = await skuReview.renameSku(oldId.trim(), newName.trim());
    } else {
      res = await skuReview.createSku(newSkuInput.value || undefined);
    }
    if (res.success) {
      addLog(res.message);
      showToastMsg(res.message, 'success');
      newSkuInput.value = '';
      await loadSkus();
    } else {
      showToastMsg(res.detail || '操作失败', 'error');
    }
  } catch (e) {
    showToastMsg('操作失败', 'error');
  }
};

const deleteSku = async () => {
  if (!confirm(`确定要删除 SKU ${selectedSkuId.value} 吗？`)) return;
  try {
    const res = await skuReview.deleteSku(selectedSkuId.value);
    if (res.success) {
      addLog(`🗑️ 已删除 SKU: ${selectedSkuId.value}`);
      showToastMsg(res.message, 'success');
      selectedSkuId.value = '';
      selectedSku.value = null;
      skuImagesData.value = [];
      await loadSkus();
    } else {
      showToastMsg(res.detail || '操作失败', 'error');
    }
  } catch (e) {
    showToastMsg('操作失败', 'error');
  }
};

// 大图查看
const openLargeImage = (url, name, event) => {
  event.stopPropagation();
  largeImageUrl.value = encodeURI(url);
  largeImageName.value = name;
  showLargeImage.value = true;
};

const closeLargeImage = () => {
  showLargeImage.value = false;
  largeImageUrl.value = '';
  largeImageName.value = '';
};

const saveDatabase = async () => {
  try {
    addLog('💾 开始保存 SKU 库更新…');
    const res = await skuReview.saveDatabase();
    if (res.success) {
      addLog('✅ SKU 库保存成功！');
      showToastMsg(res.message, 'success');
      await loadSkus();
    } else {
      showToastMsg(res.detail || '操作失败', 'error');
    }
  } catch (e) {
    showToastMsg('操作失败', 'error');
  }
};

onMounted(() => {
  addLog('🚀 系统初始化完成');
  loadFolders();
  loadSkus();
});
</script>

<style scoped>
.sku-review-page {
  padding: 20px 0;
  max-width: 100%;
  margin: 0 auto;
  background: #f5f7fa;
  min-height: 100vh;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  margin-left: 20px;
  margin-right: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 12px 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25);
}

.header h1 {
  margin: 0;
  font-size: 18px;
  color: white;
  font-weight: 600;
}

.btn-primary {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  padding: 10px 24px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
}

.btn-primary:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
}

.top-bar {
  display: flex;
  gap: 16px;
  background: white;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.sku-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.search-box input, .new-sku-box input {
  padding: 6px 10px;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  width: 160px;
  font-size: 13px;
  transition: all 0.2s;
}

.search-box input:focus, .new-sku-box input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.btn {
  padding: 6px 12px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  background: #f1f5f9;
  color: #4a5568;
  font-weight: 500;
  font-size: 13px;
  transition: all 0.2s;
}

.btn:hover:not(:disabled) {
  background: #e5e7eb;
  transform: translateY(-1px);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-sm {
  padding: 6px 14px;
  font-size: 12px;
}

.btn-success {
  background: #10b981;
  color: white;
}

.btn-success:hover:not(:disabled) {
  background: #059669;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background: #4b5563;
}

.btn-danger {
  background: #ef4444;
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: #dc2626;
}

.main-content {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  padding: 0 8px;
}

/* 文件夹导航栏 */
.folder-nav {
  width: 200px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.folder-nav-header {
  padding: 14px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.folder-nav-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.folder-nav-info {
  font-size: 12px;
  background: rgba(255, 255, 255, 0.2);
  padding: 2px 8px;
  border-radius: 10px;
}

.folder-list {
  flex: 1;
  overflow-y: auto;
  max-height: 500px;
}

.folder-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: all 0.2s;
}

.folder-item:hover {
  background: #f5f7fa;
}

.folder-item.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.folder-item.active .folder-icon {
  transform: scale(1.1);
}

.folder-icon {
  font-size: 16px;
  transition: transform 0.2s;
}

.folder-name {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.left-panel, .right-panel {
  flex: 1;
  background: white;
  padding: 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
}

.crop-count {
  font-size: 12px;
  color: #667eea;
  background: #f0f4ff;
  padding: 4px 10px;
  border-radius: 12px;
}

.panel-header {
  padding: 12px 16px;
  background: linear-gradient(135deg, #f6f8fb 0%, #eef2f7 100%);
  border-bottom: 1px solid #e0e6ed;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.panel-header h2, .panel-header h3 {
  margin: 0;
  font-size: 16px;
  color: #2c3e50;
  font-weight: 600;
}

.left-panel h2, .right-panel h2 {
  margin: 0;
  font-size: 16px;
}

.crop-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  grid-auto-rows: minmax(160px, auto);
  gap: 12px;
  padding: 20px;
  max-height: 380px;
  overflow-y: auto;
}

.crop-gallery::-webkit-scrollbar {
  width: 6px;
}

.crop-gallery::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.crop-gallery::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.crop-item {
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.2s;
  background: #f9fafb;
}

.crop-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.crop-item.selected {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.crop-item img {
  width: 100%;
  height: 130px;
  object-fit: cover;
}

.crop-name {
  font-size: 11px;
  padding: 6px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #4a5568;
  background: white;
}

.selected-crops {
  padding: 0 20px 20px;
  border-top: 1px solid #e0e6ed;
}

.selected-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 16px;
}

.selected-crops h3 {
  margin: 0;
  font-size: 14px;
  color: #2c3e50;
  font-weight: 600;
}

.selected-gallery {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 12px 0;
}

.selected-item {
  position: relative;
  width: 150px;
  height: 150px;
  cursor: pointer;
  transition: all 0.2s;
}

.selected-item:hover {
  transform: scale(1.03);
}

.selected-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14);
}

.remove-badge {
  position: absolute;
  top: -8px;
  right: -8px;
  background: #ef4444;
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
}

.sku-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  grid-auto-rows: minmax(150px, auto);
  gap: 12px;
  padding: 20px;
  max-height: 360px;
  overflow-y: auto;
}

.sku-gallery::-webkit-scrollbar {
  width: 6px;
}

.sku-gallery::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.sku-gallery::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.sku-item {
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 8px;
  overflow: hidden;
  background: #f9fafb;
  transition: all 0.2s;
}

.sku-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.sku-item.selected {
  border-color: #667eea;
  background: #f0f4ff;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.sku-cover {
  width: 100% !important;
  min-height: 85px !important;
  height: 85px !important;
  background: #e5e7eb !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-shrink: 0 !important;
}

.sku-cover img {
  width: 100% !important;
  min-height: 85px !important;
  height: 100% !important;
  object-fit: cover !important;
  flex-shrink: 0 !important;
}

.no-cover {
  font-size: 28px;
  color: #9ca3af;
}

.sku-info {
  padding: 8px 10px;
  background: white;
}

.sku-id {
  font-family: monospace;
  font-size: 12px;
  color: #667eea;
  font-weight: 600;
}

.sku-name {
  font-size: 12px;
  color: #4a5568;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin: 2px 0;
}

.sku-count {
  font-size: 11px;
  color: #9ca3af;
}

.action-hint {
  padding: 14px 20px;
  background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
  border-left: 4px solid #f59e0b;
  margin: 0 20px 16px;
  border-radius: 0 6px 6px 0;
  font-size: 13px;
  color: #92400e;
}

.action-buttons {
  display: flex;
  gap: 10px;
  padding: 0 20px 20px;
  flex-wrap: wrap;
}

.sku-detail {
  border-top: 1px solid #e0e6ed;
  padding: 0;
}

.detail-header {
  padding: 16px 20px;
  background: #f9fafb;
  border-bottom: 1px solid #e0e6ed;
}

.detail-header h3 {
  margin: 0;
  font-size: 14px;
  color: #2c3e50;
  font-weight: 600;
}

.sku-info-detail {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.sku-name-detail {
  font-size: 13px;
  color: #333;
  font-weight: 500;
}

.sku-count-detail {
  font-size: 12px;
  color: #667eea;
  background: #f0f4ff;
  padding: 2px 8px;
  border-radius: 4px;
}

.sku-detail-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
  gap: 10px;
  padding: 20px;
  max-height: 240px;
  overflow-y: auto;
}

.sku-detail-images::-webkit-scrollbar {
  width: 6px;
}

.sku-detail-images::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.sku-detail-images::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.detail-image {
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 6px;
  overflow: hidden;
  transition: all 0.2s;
  background: #f9fafb;
}

.detail-image:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.detail-image.selected {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.detail-image img {
  width: 100%;
  height: 70px;
  object-fit: cover;
}

.image-name {
  font-size: 10px;
  padding: 4px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #4a5568;
  background: white;
}

.log-panel {
  background: white;
  padding: 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  margin: 0 20px;
}

.log-panel h3 {
  font-size: 14px;
  margin: 0;
}

.logs {
  max-height: 160px;
  overflow-y: auto;
  font-size: 12px;
  font-family: monospace;
  padding: 16px 20px;
}

.logs::-webkit-scrollbar {
  width: 6px;
}

.logs::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.logs::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.log-item {
  padding: 6px 0;
  border-bottom: 1px solid #f3f4f6;
  color: #4a5568;
}

.log-item:last-child {
  border-bottom: none;
}

.toast {
  position: fixed;
  bottom: 30px;
  right: 30px;
  padding: 14px 28px;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  animation: slideIn 0.3s;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  z-index: 1000;
}

.toast.info {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.toast.success {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.toast.error {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 大图查看模态框 */
.large-image-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
  padding: 20px;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.large-image-container {
  background: white;
  border-radius: 12px;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: scaleIn 0.2s ease;
}

@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.large-image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e5e7eb;
}

.large-image-name {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
  max-width: 80%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.close-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  background: #f3f4f6;
  font-size: 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  transition: all 0.2s;
}

.close-btn:hover {
  background: #e5e7eb;
  color: #374151;
}

.large-image-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  overflow: auto;
  background: #f9fafb;
}

.large-image-wrapper img {
  max-width: 100%;
  max-height: 75vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
</style>
