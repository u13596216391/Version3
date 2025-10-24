<template>
  <div class="visualization-page">
    <div class="page-header">
      <h1>高级可视化</h1>
      <p class="subtitle">微震事件的多维度可视化分析</p>
    </div>

    <div class="tabs">
      <button 
        v-for="tab in tabs" 
        :key="tab.key"
        @click="activeTab = tab.key"
        :class="['tab-button', { active: activeTab === tab.key }]"
      >
        {{ tab.label }}
      </button>
    </div>

    <div class="tab-content">
      <Microseismic3D v-if="activeTab === '3d'" />
      <KDEWorkfaceEvolution v-if="activeTab === 'kde'" />
      <KDEGCycleEvolution v-if="activeTab === 'kdeg'" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Microseismic3D from '@/components/visualization/Microseismic3D.vue'
import KDEWorkfaceEvolution from '@/components/visualization/KDEWorkfaceEvolution.vue'
import KDEGCycleEvolution from '@/components/visualization/KDEGCycleEvolution.vue'

const activeTab = ref('3d')

const tabs = [
  { key: '3d', label: '三维散点图' },
  { key: 'kde', label: 'KDE全工作面周期演化' },
  { key: 'kdeg', label: 'KDEG全周期演化' }
]
</script>

<style scoped>
.visualization-page {
  padding: 2rem;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 2rem;
}

.page-header h1 {
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #94a3b8;
  font-size: 1rem;
}

.tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 2rem;
  border-bottom: 2px solid rgba(148, 163, 184, 0.2);
  flex-wrap: wrap;
}

.tab-button {
  padding: 0.75rem 1.5rem;
  background: transparent;
  border: none;
  color: #94a3b8;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
}

.tab-button:hover {
  color: #cbd5e1;
}

.tab-button.active {
  color: #10b981;
  border-bottom-color: #10b981;
}

.tab-content {
  background: rgba(15, 23, 42, 0.4);
  border-radius: 8px;
  min-height: 500px;
}
</style>
