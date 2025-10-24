<template>
  <div class="redirect-page p-8">
    <div class="bg-blue-50 p-6 rounded-md border border-blue-200">
      <h2 class="text-xl font-bold text-blue-800">路由已更新</h2>
      <p class="mt-2 text-blue-700">预测页面已迁移到新的“深度学习预测”模块。正在为您跳转...</p>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

onMounted(() => {
  // 自动跳转到新的深度学习预测页面
  router.replace({ path: '/predictor' })
})
</script>

<style scoped>
.redirect-page { padding-top: 2rem; }
</style>
