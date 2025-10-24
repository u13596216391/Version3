# 动画文件说明

请将以下GIF文件放置在此目录中：

## 1. KDE全工作面周期演化
文件名: `kde_workface_evolution.gif`
说明: 展示KDE（核密度估计）在整个工作面周期内的演化过程

## 2. KDEG全周期演化
文件名: `kdeg_full_cycle_evolution.gif`
说明: 展示KDEG（核密度估计梯度）在全周期内的演化过程

## 文件位置
将GIF文件直接放在当前目录下：
```
frontend/public/static/animations/
├── kde_workface_evolution.gif
└── kdeg_full_cycle_evolution.gif
```

## 访问路径
前端组件将通过以下路径访问这些文件：
- `/static/animations/kde_workface_evolution.gif`
- `/static/animations/kdeg_full_cycle_evolution.gif`

## 注意事项
- 确保文件名完全匹配（包括大小写）
- GIF文件应优化大小以确保快速加载
- 建议分辨率: 1280x720 或更高
- 建议文件大小: < 10MB
