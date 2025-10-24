<template>
  <div class="dashboard-container">
    <!-- 标题区 -->
    <div class="text-center mb-8">
      <div class="mb-4">
        <h1 class="text-4xl font-bold glow-text mb-2">矿山智能监测数据大屏</h1>
        <p class="text-tech-cyan text-lg">实时监测 · 智能预测 · 精准决策</p>
      </div>
      <div class="text-sm text-gray-400 mt-2">
        更新时间: {{ currentTime }}
      </div>
    </div>

    <!-- 核心指标卡片 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <StatCard 
        title="总任务数"
        :value="stats.total_tasks"
        icon="📊"
        color="from-blue-500 to-cyan-500"
      />
      <StatCard 
        title="运行中任务"
        :value="stats.running_tasks"
        icon="⚙️"
        color="from-purple-500 to-pink-500"
        :pulse="true"
      />
      <StatCard 
        title="已完成任务"
        :value="stats.completed_tasks"
        icon="✅"
        color="from-green-500 to-emerald-500"
      />
      <StatCard 
        title="预警总数"
        :value="alertCount"
        icon="⚠️"
        color="from-orange-500 to-red-500"
      />
    </div>

    <!-- 实时数据监控区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
      <ChartCard title="微震能量监测">
        <CompactRealtimeChart type="microseismic" />
      </ChartCard>

      <ChartCard title="支架阻力监测">
        <CompactRealtimeChart type="support_resistance" />
      </ChartCard>

      <ChartCard title="瓦斯浓度监测">
        <CompactRealtimeChart type="gas" />
      </ChartCard>

      <ChartCard title="环境温度监测">
        <CompactRealtimeChart type="temperature" />
      </ChartCard>
    </div>

    <!-- 图表区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      <!-- 任务趋势图 -->
      <ChartCard title="任务执行趋势">
        <TaskTrendChart :data="taskTrend" />
      </ChartCard>

      <!-- 预警分布图 -->
      <ChartCard title="预警级别分布">
        <AlertDistributionChart :data="alertStats" />
      </ChartCard>
    </div>

    <!-- 底部区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 最近预警列表 -->
      <div class="lg:col-span-2">
        <ChartCard title="最近预警">
          <AlertList :alerts="recentAlerts" />
        </ChartCard>
      </div>

      <!-- 系统状态 -->
      <ChartCard title="系统状态">
        <SystemStatus :health="stats.system_health" />
      </ChartCard>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import StatCard from '@/components/dashboard/StatCard.vue'
import ChartCard from '@/components/dashboard/ChartCard.vue'
import TaskTrendChart from '@/components/dashboard/TaskTrendChart.vue'
import AlertDistributionChart from '@/components/dashboard/AlertDistributionChart.vue'
import AlertList from '@/components/dashboard/AlertList.vue'
import SystemStatus from '@/components/dashboard/SystemStatus.vue'
import CompactRealtimeChart from '@/components/dashboard/CompactRealtimeChart.vue'

const currentTime = ref('')
const stats = ref({
  total_tasks: 0,
  running_tasks: 0,
  completed_tasks: 0,
  failed_tasks: 0,
  system_health: 'normal'
})
const recentAlerts = ref([])
const alertStats = ref([])
const taskTrend = ref([])
const alertCount = ref(0)

let refreshInterval = null

// 更新当前时间
const updateTime = () => {
  const now = new Date()
  currentTime.value = now.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

// 获取大屏数据
const fetchDashboardData = async () => {
  try {
    const response = await axios.get('/api/dashboard/overview/')
    const data = response.data
    
    stats.value = data.overview || stats.value
    recentAlerts.value = data.recent_alerts || []
    alertStats.value = data.alert_stats || []
    taskTrend.value = data.task_trend || []
    alertCount.value = recentAlerts.value.length
  } catch (error) {
    console.error('获取大屏数据失败:', error)
  }
}

onMounted(() => {
  updateTime()
  fetchDashboardData()
  
  // 每秒更新时间
  setInterval(updateTime, 1000)
  
  // 每5秒刷新数据
  refreshInterval = setInterval(fetchDashboardData, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.dashboard-container {
  @apply min-h-screen;
}
</style>
