<template>
  <div class="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
    <div class="flex justify-between items-center mb-6 pb-6 border-b border-gray-700">
      <h2 class="text-2xl font-bold text-white">训练结果分析</h2>
      <button 
        @click="$emit('reset')" 
        class="px-6 py-2 bg-gray-700 text-gray-200 font-semibold rounded-lg hover:bg-gray-600 transition-colors"
      >
        ← 返回配置
      </button>
    </div>
    
    <div class="space-y-8">
      <!-- 多模型对比概览 -->
      <div v-if="results.length > 1" class="p-6 bg-gradient-to-r from-violet-900/30 to-purple-900/30 rounded-lg border border-violet-700">
        <h3 class="font-bold text-lg text-gray-200 mb-4">📊 模型性能对比</h3>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div v-for="result in results" :key="result.model_name" class="bg-gray-700 p-3 rounded-md shadow-sm text-center">
            <p class="text-xs font-medium text-gray-400 mb-1">{{ result.model_name }}</p>
            <p class="text-2xl font-bold text-violet-400">{{ result.metrics.MSE?.toFixed(4) || 'N/A' }}</p>
            <p class="text-xs text-gray-500">MSE</p>
          </div>
        </div>
      </div>

      <!-- 每个模型的详细结果 -->
      <div v-for="result in results" :key="result.model_name" class="p-6 border-2 border-gray-700 rounded-lg hover:border-violet-500 transition-colors bg-gray-900/30">
        <!-- 模型标题 -->
        <div class="flex items-center justify-between mb-4">
          <h3 class="font-bold text-xl text-violet-400">{{ result.model_name }}</h3>
          <span class="px-3 py-1 bg-green-900/50 text-green-400 text-xs font-semibold rounded-full border border-green-700">
            ✓ 训练完成
          </span>
        </div>

        <!-- 评估指标 -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          <div v-for="(value, key) in result.metrics" :key="key" class="metric-card">
            <p class="metric-label">{{ key }}</p>
            <p class="metric-value">{{ formatMetric(value) }}</p>
          </div>
        </div>

        <!-- 预测结果图表 -->
        <div v-if="result.plot_data" class="mt-6">
          <InteractiveChart 
            :title="`${result.model_name} - 预测对比`"
            :plotData="result.plot_data" 
          />
        </div>
        <div v-else class="text-center text-gray-500 py-8 bg-gray-700/30 rounded-md">
          <svg class="mx-auto h-12 w-12 text-gray-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p>此模型未生成可视化数据</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import InteractiveChart from '@/components/common/InteractiveChart.vue'

defineProps({
  results: {
    type: Array,
    required: true
  }
})

defineEmits(['reset'])

const formatMetric = (value) => {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value === 'number') {
    return value.toFixed(4)
  }
  return value
}
</script>

<style scoped>
.metric-card {
  @apply p-4 bg-gray-700 rounded-md shadow-sm text-center border border-gray-600 hover:shadow-md transition-shadow;
}

.metric-label {
  @apply text-xs text-gray-400 font-medium uppercase tracking-wide mb-1;
}

.metric-value {
  @apply text-lg font-bold text-violet-400;
}
</style>
