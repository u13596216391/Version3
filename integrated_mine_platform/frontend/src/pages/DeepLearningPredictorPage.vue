<template>
  <div class="predictor-page">
    <h1 class="text-3xl font-bold text-gray-800 mb-6">深度学习预测模块</h1>
    <p class="text-gray-600 mb-8">使用深度学习模型进行微震能量和支架阻力的时间序列预测</p>
    
    <!-- 根据当前视图，条件性地渲染子组件 -->
    <div v-if="currentView === 'config'">
      <DL_ConfigPanel @start-training="handleStartTraining" />
    </div>
    
    <div v-else-if="currentView === 'monitoring'">
      <TaskMonitor 
        :taskId="taskId" 
        :apiEndpoints="apiEndpoints" 
        @task-complete="handleTaskComplete" 
        @task-failed="handleTaskFailed"
      />
    </div>

    <div v-else-if="currentView === 'results'">
      <DL_ResultsDisplay :results="resultsData" @reset="resetPage" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import DL_ConfigPanel from '@/components/dl_predictor/DL_ConfigPanel.vue'
import TaskMonitor from '@/components/common/TaskMonitor.vue'
import DL_ResultsDisplay from '@/components/dl_predictor/DL_ResultsDisplay.vue'

// 定义API端点
const apiEndpoints = {
  status: '/api/predictor/status/',
  results: '/api/predictor/results/'
}

// 响应式状态
const currentView = ref('config') // 'config', 'monitoring', 'results'
const taskId = ref(null)
const resultsData = ref(null)

// 处理函数
const handleStartTraining = (id) => {
  taskId.value = id
  currentView.value = 'monitoring'
}

const handleTaskComplete = (results) => {
  resultsData.value = results
  currentView.value = 'results'
}

const handleTaskFailed = () => {
  alert('❌ 训练任务失败，请检查后端日志或重新配置参数！')
  resetPage()
}

const resetPage = () => {
  currentView.value = 'config'
  taskId.value = null
  resultsData.value = null
}
</script>

<style scoped>
.predictor-page {
  @apply min-h-screen p-8;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
</style>
