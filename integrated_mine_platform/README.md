# 集成矿山智能预测平台

一个集成了支架阻力预测、微震预测和实时监控的智能矿山管理平台。

## 技术栈

### 后端
- Django 5.0
- Django REST Framework
- SQLite数据库

### 前端
- Vue 3 (Composition API)
- Vite
- TailwindCSS
- ECharts (数据可视化)
- Axios

## 功能特性

### 1. 数据大屏
- 实时显示系统统计数据
- 任务执行趋势图
- 预警级别分布
- 系统运行状态监控

### 2. 支架阻力预测
- 多模型并行训练
- 数据上传和预处理
- 预测结果可视化

### 3. 微震预测
- 微震能量预测
- 时间序列分析
- 预测结果展示

### 4. 实时监控
- 微震能量实时监测
- 支架阻力实时监测
- 瓦斯浓度监测
- 环境温度监测

### 5. 数据查看
- 数据库表概览
- 数据记录查询
- 统计信息展示

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+

### 安装步骤

#### 1. 克隆项目
```bash
cd integrated_mine_platform
```

#### 2. 后端安装
```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 数据库迁移
python manage.py makemigrations
python manage.py migrate

# 创建超级用户（可选）
python manage.py createsuperuser
```

#### 3. 前端安装
```bash
cd frontend
npm install
```

#### 4. 启动服务

##### 方式一：手动启动（推荐开发）

**终端1 - 启动后端：**
```bash
python manage.py runserver
```
后端将在 http://127.0.0.1:8000 运行

**终端2 - 启动前端：**
```bash
cd frontend
npm run dev
```
前端将在 http://localhost:5173 运行

##### 方式二：使用启动脚本（Windows）
```bash
.\start_development.bat
```

### 访问应用
打开浏览器访问：http://localhost:5173

### 管理后台
访问：http://127.0.0.1:8000/admin

## 项目结构

```
integrated_mine_platform/
├── integrated_mine/          # Django项目配置
│   ├── settings.py           # 项目设置
│   ├── urls.py              # 主路由
│   └── wsgi.py              # WSGI配置
├── dashboard_app/            # 数据大屏应用
│   ├── models.py            # 数据模型
│   ├── views.py             # API视图
│   └── urls.py              # 路由配置
├── monitoring_app/           # 实时监控应用
├── predictor_app/            # 支架阻力预测（待集成）
├── microseismic_app/         # 微震预测（待集成）
├── frontend/                 # Vue 3前端
│   ├── src/
│   │   ├── components/      # 可复用组件
│   │   │   ├── dashboard/   # 大屏组件
│   │   │   └── monitoring/  # 监控组件
│   │   ├── pages/           # 页面组件
│   │   ├── router/          # 路由配置
│   │   ├── App.vue          # 根组件
│   │   └── main.js          # 入口文件
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── media/                    # 媒体文件
├── manage.py                 # Django管理脚本
├── requirements.txt          # Python依赖
└── README.md
```

## 设计风格

### 深蓝色科技风格主题
- 主色调：深蓝色 (#0056e0)
- 背景：深色科技渐变
- 强调色：科技蓝 (#0066ff)、青色 (#00d4ff)
- 卡片：毛玻璃效果 + 渐变边框
- 动画：流畅的过渡和发光效果

## API文档

### 数据大屏
- `GET /api/dashboard/overview/` - 获取大屏总览数据
- `POST /api/dashboard/statistics/update/` - 更新统计数据
- `GET /api/dashboard/alerts/` - 获取预警列表
- `GET /api/dashboard/data-view/` - 获取数据库概览

### 实时监控
- `GET /api/monitoring/realtime/` - 获取实时监控数据
- `POST /api/monitoring/realtime/` - 上传监控数据

## 开发说明

### 添加新功能
1. 后端：在对应app的views.py中添加API
2. 前端：在pages/或components/中添加Vue组件
3. 更新路由：在router/index.js中注册新路由

### 样式规范
- 使用TailwindCSS工具类
- 遵循深蓝色科技风格
- 使用自定义CSS类（.tech-card, .tech-button等）

### 代码提交
- 遵循规范的commit message
- 测试后再提交
- 保持代码整洁

## 后续开发计划

1. ✅ 完成基础框架搭建
2. ✅ 实现数据大屏
3. ⏳ 集成支架阻力预测算法
4. ⏳ 集成微震预测算法
5. ⏳ 完善实时监控功能
6. ⏳ 添加用户认证系统
7. ⏳ 性能优化和测试

## 许可证

MIT License

## 联系方式

如有问题，请联系开发团队。
