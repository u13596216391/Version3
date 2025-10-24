<template>
  <div class="analysis-page">
    <div class="page-header">
      <h1 class="title">多源数据分析</h1>
      <p class="subtitle">微震散点图、核密度图、支架阻力DWT分析</p>
    </div>

    <div class="analysis-container">
      <!-- 分析类型选择 -->
      <div class="analysis-tabs">
        <button 
          v-for="tab in tabs" 
          :key="tab.key"
          :class="['tab-btn', { active: activeTab === tab.key }]"
          @click="activeTab = tab.key"
        >
          {{ tab.label }}
        </button>
      </div>

      <!-- 微震散点图 -->
      <div v-if="activeTab === 'microseismic-scatter'" class="analysis-content">
        <MicroseismicScatter />
      </div>

      <!-- 微震核密度图 -->
      <div v-else-if="activeTab === 'microseismic-density'" class="analysis-content">
        <MicroseismicDensity />
      </div>

      <!-- 支架阻力DWT分析 -->
      <div v-else-if="activeTab === 'support-dwt'" class="analysis-content">
        <SupportDWTAnalysis />
      </div>

      <!-- 支架阻力小波对比 -->
      <div v-else-if="activeTab === 'wavelet-comparison'" class="analysis-content">
        <WaveletComparison />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import MicroseismicScatter from '../components/analysis/MicroseismicScatter.vue'
import MicroseismicDensity from '../components/analysis/MicroseismicDensity.vue'
import SupportDWTAnalysis from '../components/analysis/SupportDWTAnalysis.vue'
import WaveletComparison from '../components/analysis/WaveletComparison.vue'

const activeTab = ref('microseismic-scatter')

const tabs = [
  { key: 'microseismic-scatter', label: '微震散点图' },
  { key: 'microseismic-density', label: '微震核密度图' },
  { key: 'support-dwt', label: '支架阻力DWT分析' },
  { key: 'wavelet-comparison', label: '小波对比分析' }
]
</script>

<style scoped>
.analysis-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  padding: 2rem;
}

.page-header {
  text-align: center;
  margin-bottom: 2rem;
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #94a3b8;
  font-size: 1rem;
}

.analysis-container {
  max-width: 1400px;
  margin: 0 auto;
}

.analysis-tabs {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  background: rgba(30, 41, 59, 0.8);
  padding: 1rem;
  border-radius: 12px;
  backdrop-filter: blur(10px);
}

.tab-btn {
  flex: 1;
  padding: 0.75rem 1.5rem;
  background: rgba(51, 65, 85, 0.6);
  border: none;
  border-radius: 8px;
  color: #cbd5e1;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-btn:hover {
  background: rgba(71, 85, 105, 0.8);
  transform: translateY(-2px);
}

.tab-btn.active {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.analysis-content {
  background: rgba(30, 41, 59, 0.8);
  border-radius: 12px;
  padding: 2rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
</style>
