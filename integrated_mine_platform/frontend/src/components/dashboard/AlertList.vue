<template>
  <div class="alert-list max-h-[300px] overflow-y-auto">
    <div v-if="alerts.length === 0" class="text-center text-gray-400 py-8">
      暂无预警信息
    </div>
    <div v-else class="space-y-3">
      <div 
        v-for="alert in alerts" 
        :key="alert.id"
        :class="['alert-item', `alert-${alert.level}`]"
      >
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <div class="flex items-center space-x-2 mb-1">
              <span class="alert-badge">{{ getLevelText(alert.level) }}</span>
              <span class="text-sm text-gray-400">{{ alert.source }}</span>
            </div>
            <h4 class="font-semibold mb-1">{{ alert.title }}</h4>
            <p class="text-sm text-gray-300">{{ alert.message }}</p>
          </div>
          <div class="text-xs text-gray-500 ml-4 whitespace-nowrap">
            {{ formatTime(alert.timestamp) }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  alerts: {
    type: Array,
    default: () => []
  }
})

const getLevelText = (level) => {
  const map = {
    info: '信息',
    warning: '警告',
    danger: '危险',
    critical: '严重'
  }
  return map[level] || level
}

const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date
  
  if (diff < 60000) {
    return '刚刚'
  } else if (diff < 3600000) {
    return `${Math.floor(diff / 60000)}分钟前`
  } else if (diff < 86400000) {
    return `${Math.floor(diff / 3600000)}小时前`
  } else {
    return date.toLocaleDateString('zh-CN')
  }
}
</script>

<style scoped>
.alert-item {
  @apply p-4 rounded-lg border-l-4 transition-all duration-200;
  background: rgba(10, 14, 39, 0.5);
}

.alert-item:hover {
  @apply transform -translate-y-1 shadow-lg;
}

.alert-info {
  @apply border-blue-500;
}

.alert-warning {
  @apply border-yellow-500;
}

.alert-danger {
  @apply border-orange-500;
}

.alert-critical {
  @apply border-red-500;
}

.alert-badge {
  @apply px-2 py-1 rounded text-xs font-semibold;
}

.alert-info .alert-badge {
  @apply bg-blue-500/20 text-blue-400;
}

.alert-warning .alert-badge {
  @apply bg-yellow-500/20 text-yellow-400;
}

.alert-danger .alert-badge {
  @apply bg-orange-500/20 text-orange-400;
}

.alert-critical .alert-badge {
  @apply bg-red-500/20 text-red-400;
}
</style>
