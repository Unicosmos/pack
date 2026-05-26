<template>
  <main 
    class="page-container" 
    :class="{ 
      'has-sidebar': hasSidebar,
      'sidebar-left': sidebarPosition === 'left',
      'sidebar-right': sidebarPosition === 'right'
    }"
  >
    <aside class="sidebar" v-if="hasSidebar">
      <slot name="sidebar">
        <div class="sidebar-content">
          <slot name="sidebar-header"></slot>
          <slot name="sidebar-body"></slot>
          <slot name="sidebar-footer"></slot>
        </div>
      </slot>
    </aside>
    
    <div class="main-content">
      <div class="content-wrapper">
        <slot name="header">
          <div class="content-header" v-if="showHeader && (title || subtitle)">
            <div v-if="title" class="content-title">{{ title }}</div>
            <div v-if="subtitle" class="content-subtitle">{{ subtitle }}</div>
          </div>
        </slot>
        
        <div class="content-body">
          <slot></slot>
        </div>
        
        <div class="content-footer" v-if="$slots.footer">
          <slot name="footer"></slot>
        </div>
      </div>
    </div>
    
    <aside class="sidebar sidebar-right" v-if="hasSidebar && sidebarPosition === 'right'">
      <slot name="sidebar-right"></slot>
    </aside>
  </main>
</template>

<script setup>
import { computed, useSlots } from 'vue'

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  subtitle: {
    type: String,
    default: ''
  },
  showHeader: {
    type: Boolean,
    default: false
  },
  hasSidebar: {
    type: Boolean,
    default: false
  },
  sidebarPosition: {
    type: String,
    default: 'left',
    validator: (value) => ['left', 'right'].includes(value)
  },
  maxWidth: {
    type: String,
    default: '1800px'
  },
  padding: {
    type: String,
    default: '20px'
  }
})

const slots = useSlots()
const hasSidebar = computed(() => props.hasSidebar || !!slots.sidebar || !!slots['sidebar-right'])
</script>

<style scoped>
.page-container {
  min-height: calc(100vh - 80px);
  background: var(--color-bg-secondary);
  display: flex;
  transition: all 0.3s ease;
}

.main-content {
  flex: 1;
  display: flex;
  justify-content: center;
  padding: 20px;
  transition: all 0.3s ease;
}

.content-wrapper {
  width: 100%;
  max-width: 1800px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.content-header {
  background: var(--color-bg-primary);
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
}

.content-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 6px 0;
}

.content-subtitle {
  font-size: 14px;
  color: var(--color-text-secondary);
  margin: 0;
}

.content-body {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.content-footer {
  padding-top: 16px;
  border-top: 1px solid var(--color-border);
}

.sidebar {
  width: 280px;
  flex-shrink: 0;
  padding: 20px;
  background: var(--color-bg-primary);
  border-right: 1px solid var(--color-border);
  transition: all 0.3s ease;
}

.sidebar-right {
  border-right: none;
  border-left: 1px solid var(--color-border);
}

.sidebar-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.page-container.sidebar-left .main-content {
  padding-left: 0;
}

.page-container.sidebar-right .main-content {
  padding-right: 0;
}

@media (max-width: 1200px) {
  .sidebar {
    width: 240px;
  }
  
  .main-content {
    padding: 16px;
  }
}

@media (max-width: 1024px) {
  .page-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--color-border);
    padding: 16px;
  }
  
  .sidebar-right {
    border-left: none;
    border-top: 1px solid var(--color-border);
    border-bottom: none;
  }
  
  .main-content {
    padding: 16px;
  }
  
  .page-container.sidebar-left .main-content,
  .page-container.sidebar-right .main-content {
    padding-left: 16px;
    padding-right: 16px;
  }
}

@media (max-width: 768px) {
  .main-content {
    padding: 12px;
  }
  
  .content-wrapper {
    gap: 12px;
  }
  
  .content-header {
    padding: 12px 16px;
  }
  
  .content-title {
    font-size: 16px;
  }
  
  .content-subtitle {
    font-size: 13px;
  }
}

@media (max-width: 480px) {
  .main-content {
    padding: 8px;
  }
  
  .content-header {
    padding: 10px 14px;
  }
}
</style>