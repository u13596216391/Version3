<template>
  <div class="tech-card h-full">
    <h3 class="text-xl font-bold mb-4 text-tech-cyan">{{ title }}</h3>
    <div class="chart-content">
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: String
})
</script>

<style scoped>
.chart-content {
  @apply min-h-[300px] flex items-center justify-center;
}
</style>
