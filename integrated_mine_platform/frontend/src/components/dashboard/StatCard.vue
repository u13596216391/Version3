<template>
  <div class="tech-card hover:shadow-glow transition-all duration-300">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center space-x-3">
        <div :class="['icon-container', colorClass]">
          <span class="text-2xl">{{ icon }}</span>
        </div>
        <div>
          <p class="text-gray-400 text-sm">{{ title }}</p>
          <h3 :class="['text-3xl font-bold', pulse ? 'animate-pulse-slow' : '']">
            {{ formattedValue }}
          </h3>
        </div>
      </div>
    </div>
    <div class="h-1 rounded-full overflow-hidden bg-gray-800">
      <div 
        :class="['h-full bg-gradient-to-r transition-all duration-500', colorClass]"
        :style="{ width: '100%' }"
      ></div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: String,
  value: {
    type: Number,
    default: 0
  },
  icon: String,
  color: {
    type: String,
    default: 'from-blue-500 to-cyan-500'
  },
  pulse: {
    type: Boolean,
    default: false
  }
})

const formattedValue = computed(() => {
  return props.value.toLocaleString()
})

const colorClass = computed(() => props.color)
</script>

<style scoped>
.icon-container {
  @apply w-14 h-14 rounded-lg flex items-center justify-center;
  @apply bg-gradient-to-br shadow-lg;
}
</style>
