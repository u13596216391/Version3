import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('@/pages/DashboardPage.vue')
  },
  {
    path: '/predictor',
    name: 'Predictor',
    component: () => import('@/pages/PredictorPage.vue')
  },
  {
    path: '/microseismic',
    name: 'Microseismic',
    component: () => import('@/pages/MicroseismicPage.vue')
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
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
