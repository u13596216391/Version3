<template>
  <div class="min-h-screen bg-tech-gradient">
    <!-- 导航栏 -->
    <nav class="bg-tech-darker/90 backdrop-blur-md border-b border-primary-500/30 sticky top-0 z-50">
      <div class="container mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <!-- Logo -->
          <div class="flex items-center space-x-4">
            <div class="w-10 h-10 bg-gradient-to-br from-primary-500 to-tech-cyan rounded-lg flex items-center justify-center">
              <span class="text-2xl font-bold">矿</span>
            </div>
            <div>
              <h1 class="text-xl font-bold glow-text">集成矿山智能预测平台</h1>
              <p class="text-xs text-tech-cyan">Integrated Mine Intelligence Platform</p>
            </div>
          </div>
          
          <!-- 导航菜单 -->
          <div class="flex space-x-6">
            <router-link 
              v-for="route in routes" 
              :key="route.path"
              :to="route.path"
              class="nav-link"
            >
              {{ route.name }}
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mx-auto px-6 py-8">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- 页脚 -->
    <footer class="bg-tech-darker/50 border-t border-primary-500/20 mt-auto">
      <div class="container mx-auto px-6 py-4 text-center text-sm text-gray-400">
        <p>&copy; 2024 集成矿山智能预测平台 | Powered by Vue 3 + Django</p>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const routes = [
  { path: '/', name: '数据大屏' },
  { path: '/predictor', name: '支架阻力预测' },
  { path: '/microseismic', name: '微震预测' },
  { path: '/monitoring', name: '实时监控' },
  { path: '/data-view', name: '数据查看' },
]
</script>

<style scoped>
.nav-link {
  @apply text-gray-300 hover:text-tech-cyan transition-colors duration-200 
         font-medium relative;
}

.nav-link::after {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(90deg, #0066ff, #00d4ff);
  transition: width 0.3s ease;
}

.nav-link:hover::after,
.nav-link.router-link-active::after {
  width: 100%;
}

.nav-link.router-link-active {
  @apply text-tech-cyan;
}

/* 页面切换动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
