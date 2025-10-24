<template>
  <div class="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 text-center max-w-2xl mx-auto">
    <h2 class="text-2xl font-bold text-white mb-6">任务监控</h2>
    <div class="space-y-6">
      <!-- 状态图标 -->
      <div class="flex justify-center items-center">
        <div v-if="isRunning" class="w-16 h-16">
          <svg class="animate-spin h-full w-full text-violet-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        </div>
        <div v-if="isSuccess" class="w-16 h-16 text-green-400">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-full h-full">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
        </div>
        <div v-if="isFailed" class="w-16 h-16 text-red-400">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-full h-full">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
        </div>
      </div>

      <!-- 状态信息 -->
      <p class="text-lg font-semibold" :class="statusColor">{{ statusMessage }}</p>
      <p class="text-gray-400 text-sm">任务ID: <span class="font-mono bg-gray-700 p-1 rounded-md">{{ taskId }}</span></p>

      <!-- 进度条 -->
      <div class="w-full bg-gray-700 rounded-full h-2.5">
        <div class="bg-gradient-to-r from-purple-400 to-violet-600 h-2.5 rounded-full transition-all duration-500" :style="{ width: progress + '%' }"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import axios from 'axios'

const props = defineProps({
  taskId: String,
  apiEndpoints: Object
})

const emit = defineEmits(['task-complete', 'task-failed'])

const status = ref('PENDING') // PENDING, RUNNING, SUCCESS, FAILED
const statusMessage = ref('正在初始化...')
const progress = ref(0)
let pollInterval = null

const isRunning = computed(() => status.value === 'RUNNING' || status.value === 'PENDING')
const isSuccess = computed(() => status.value === 'SUCCESS')
const isFailed = computed(() => status.value === 'FAILED')

const statusColor = computed(() => {
  if (isFailed.value) return 'text-red-400'
  if (isSuccess.value) return 'text-green-400'
  return 'text-gray-200'
})

const pollStatus = async () => {
  try {
    const response = await axios.get(`${props.apiEndpoints.status}${props.taskId}/`)
    const data = response.data
    
    status.value = data.status
    statusMessage.value = data.message

    if (data.status === 'RUNNING') {
      const match = data.message.match(/\((\d+)\/(\d+)\)/) || data.message.match(/Epoch (\d+)\/(\d+)/)
      if (match) {
        progress.value = (parseInt(match[1]) / parseInt(match[2])) * 100
      } else if (progress.value < 90) {
        // 如果没有明确进度，就缓慢增加
        progress.value += 2
      }
    } else if (data.status === 'SUCCESS') {
      progress.value = 100
      statusMessage.value = '任务成功完成！正在获取结果...'
      clearInterval(pollInterval)
      fetchResults()
    } else if (data.status === 'FAILED') {
      statusMessage.value = `任务失败: ${data.error || '未知错误'}`
      clearInterval(pollInterval)
      emit('task-failed')
    }
  } catch (error) {
    console.error('轮询状态失败:', error)
    statusMessage.value = '轮询状态失败，请检查网络连接。'
    status.value = 'FAILED'
    clearInterval(pollInterval)
    emit('task-failed')
  }
}

const fetchResults = async () => {
  try {
    const response = await axios.get(`${props.apiEndpoints.results}${props.taskId}/`)
    emit('task-complete', response.data)
  } catch (error) {
    console.error('获取结果失败:', error)
    statusMessage.value = '获取最终结果失败！'
    status.value = 'FAILED'
    emit('task-failed')
  }
}

onMounted(() => {
  pollStatus()
  pollInterval = setInterval(pollStatus, 3000)
})

onUnmounted(() => {
  clearInterval(pollInterval)
})
</script>
