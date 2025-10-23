<template>
  <div class="system-status">
    <div class="status-indicator mb-6">
      <div :class="['status-circle', healthClass]"></div>
      <div class="ml-4">
        <h3 class="text-2xl font-bold">{{ healthText }}</h3>
        <p class="text-sm text-gray-400">系统运行状态</p>
      </div>
    </div>

    <div class="space-y-4">
      <div class="status-item">
        <div class="flex justify-between items-center mb-2">
          <span class="text-gray-300">CPU使用率</span>
          <span class="font-semibold text-tech-cyan">{{ cpuUsage }}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill bg-tech-cyan" :style="{ width: `${cpuUsage}%` }"></div>
        </div>
      </div>

      <div class="status-item">
        <div class="flex justify-between items-center mb-2">
          <span class="text-gray-300">内存使用率</span>
          <span class="font-semibold text-primary-400">{{ memoryUsage }}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill bg-primary-400" :style="{ width: `${memoryUsage}%` }"></div>
        </div>
      </div>

      <div class="status-item">
        <div class="flex justify-between items-center mb-2">
          <span class="text-gray-300">数据库连接</span>
          <span class="font-semibold text-green-400">正常</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill bg-green-400" style="width: 100%"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'

const props = defineProps({
  health: {
    type: String,
    default: 'normal'
  }
})

const cpuUsage = ref(45)
const memoryUsage = ref(62)

const healthClass = computed(() => {
  const classes = {
    normal: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500'
  }
  return classes[props.health] || classes.normal
})

const healthText = computed(() => {
  const texts = {
    normal: '运行正常',
    warning: '需要注意',
    error: '存在异常'
  }
  return texts[props.health] || texts.normal
})

// 模拟动态更新
onMounted(() => {
  setInterval(() => {
    cpuUsage.value = Math.floor(Math.random() * 30) + 30
    memoryUsage.value = Math.floor(Math.random() * 20) + 50
  }, 3000)
})
</script>

<style scoped>
.status-indicator {
  @apply flex items-center;
}

.status-circle {
  @apply w-16 h-16 rounded-full animate-pulse-slow;
  box-shadow: 0 0 20px currentColor;
}

.status-item {
  @apply p-3 rounded-lg;
  background: rgba(0, 51, 102, 0.2);
}

.progress-bar {
  @apply h-2 bg-gray-800 rounded-full overflow-hidden;
}

.progress-fill {
  @apply h-full rounded-full transition-all duration-500;
}
</style>
