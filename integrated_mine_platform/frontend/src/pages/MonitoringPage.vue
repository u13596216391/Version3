<template>
  <div class="monitoring-page">
    <div class="tech-card mb-6">
      <h2 class="text-2xl font-bold glow-text mb-4">实时监控</h2>
      <p class="text-gray-400">实时监测矿山各项指标数据</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <ChartCard title="微震能量监测">
        <RealtimeChart type="microseismic" />
      </ChartCard>

      <ChartCard title="支架阻力监测">
        <RealtimeChart type="support_resistance" />
      </ChartCard>

      <ChartCard title="瓦斯浓度监测">
        <RealtimeChart type="gas" />
      </ChartCard>

      <ChartCard title="环境温度监测">
        <RealtimeChart type="temperature" />
      </ChartCard>
    </div>
  </div>
</template>

<script setup>
import ChartCard from '@/components/dashboard/ChartCard.vue'
import RealtimeChart from '@/components/monitoring/RealtimeChart.vue'
</script>
