<template>
  <div class="kdeg-evolution">
    <div class="gif-container">
      <h3>KDEG全周期演化</h3>
      <div class="image-wrapper">
        <img 
          v-if="gifExists" 
          :src="gifPath" 
          alt="KDEG全周期演化"
          class="evolution-gif"
        />
        <div v-else class="placeholder">
          <div class="placeholder-icon">📈</div>
          <p>GIF文件未找到</p>
          <p class="hint">请将GIF文件放置在: <code>{{ gifPath }}</code></p>
        </div>
      </div>
      <div class="description">
        <p>本图展示了KDEG（核密度估计梯度）在全周期内的演化过程，揭示微震活动强度和分布梯度的时空演化特征。</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const gifPath = '/static/animations/kdeg_full_cycle_evolution.gif'
const gifExists = ref(false)

onMounted(() => {
  // 检查GIF文件是否存在
  const img = new Image()
  img.onload = () => {
    gifExists.value = true
  }
  img.onerror = () => {
    gifExists.value = false
  }
  img.src = gifPath
})
</script>

<style scoped>
.kdeg-evolution {
  padding: 1.5rem;
}

.gif-container {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 1.5rem;
}

h3 {
  color: #e2e8f0;
  font-size: 1.5rem;
  margin-bottom: 1.5rem;
  text-align: center;
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.image-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 500px;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 6px;
  overflow: hidden;
}

.evolution-gif {
  max-width: 100%;
  max-height: 800px;
  width: auto;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  color: #94a3b8;
  text-align: center;
}

.placeholder-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  opacity: 0.5;
}

.placeholder p {
  margin: 0.5rem 0;
  font-size: 1rem;
}

.placeholder .hint {
  font-size: 0.875rem;
  color: #64748b;
  margin-top: 1rem;
}

.placeholder code {
  background: rgba(51, 65, 85, 0.6);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  color: #f472b6;
}

.description {
  margin-top: 1.5rem;
  padding: 1rem;
  background: rgba(51, 65, 85, 0.4);
  border-radius: 6px;
  border-left: 4px solid #f093fb;
}

.description p {
  color: #cbd5e1;
  line-height: 1.6;
  margin: 0;
}
</style>
