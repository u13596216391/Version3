import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('@/pages/DashboardPage.vue')
  },
  {
    path: '/predictor',
    name: 'DeepLearningPredictor',
    component: () => import('@/pages/DeepLearningPredictorPage.vue')
  },
  {
    path: '/monitoring',
    name: 'Monitoring',
    component: () => import('@/pages/MonitoringPage.vue')
  },
  {
    path: '/data-view',
    name: 'DataView',
    component: () => import('@/pages/DataViewPage.vue')
  },
  {
    path: '/analysis',
    name: 'Analysis',
    component: () => import('@/pages/AnalysisPage.vue')
  },
  {
    path: '/visualization',
    name: 'Visualization',
    component: () => import('@/pages/VisualizationPage.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
