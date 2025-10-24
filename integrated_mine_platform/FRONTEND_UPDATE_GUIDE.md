# 前端更新未应用问题解决方案

## 🔍 问题诊断

### 原因
前端使用了 **Docker Multi-Stage Build**，源代码在构建时编译为静态文件：
1. **构建阶段**: Vue源码 → 编译 → 静态文件 (`/app/dist`)
2. **生产阶段**: 静态文件 → 复制到 Nginx → 部署

当您修改Vue组件（如 `MicroseismicScatter.vue`）后：
- ✅ 源文件已更新
- ❌ Docker镜像中的**编译后静态文件**仍是旧版本
- ❌ `docker-compose restart` 只重启容器，不重新构建镜像

## ✅ 解决步骤

### 方法1: 重新构建前端镜像（推荐）

```bash
# 1. 无缓存重新构建前端
docker-compose build --no-cache frontend

# 2. 重启前端容器
docker-compose up -d frontend

# 3. 清理浏览器缓存并刷新页面
# Chrome/Edge: Ctrl + Shift + R (硬刷新)
# Firefox: Ctrl + F5
```

### 方法2: 完整重建（如果方法1不生效）

```bash
# 停止并删除容器
docker-compose down

# 删除前端镜像
docker rmi integrated_mine_platform-frontend

# 重新构建并启动
docker-compose up -d --build
```

### 方法3: 开发模式（开发调试时使用）

修改 `docker-compose.yml`，添加volume挂载实现热重载：

```yaml
frontend:
  build: ./frontend
  ports:
    - "5173:5173"  # Vite开发服务器
  volumes:
    - ./frontend:/app
    - /app/node_modules
  command: npm run dev
  environment:
    - VITE_API_URL=http://backend:8000
```

## 📋 验证更新

### 1. 检查前端日志
```bash
docker-compose logs frontend --tail 20
```

应该看到新的构建时间戳。

### 2. 浏览器开发者工具
- 打开 Network 标签
- 刷新页面
- 查看 `AnalysisPage-*.js` 文件的哈希值是否变化
- 新构建的文件会有新的哈希值（如 `AnalysisPage-XYZ123.js`）

### 3. 测试新功能
访问 http://localhost/analysis，应该看到：
- ✅ "数据来源" 选择器
- ✅ "选择数据集" 下拉菜单
- ✅ 数据集自动加载

## 🚀 快速命令参考

```bash
# 查看前端容器状态
docker-compose ps frontend

# 查看前端日志
docker-compose logs frontend --tail 50

# 重新构建前端（无缓存）
docker-compose build --no-cache frontend

# 重启前端容器
docker-compose restart frontend

# 重建并启动前端
docker-compose up -d --build frontend

# 进入前端容器检查文件
docker-compose exec frontend sh
ls -la /usr/share/nginx/html/assets/
```

## 🔄 未来更新最佳实践

### 后端更新（Python代码）
```bash
# 后端支持热重载，只需重启
docker-compose restart backend
```

### 前端更新（Vue组件）
```bash
# 前端需要重新构建
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

### 完整更新（前后端都改了）
```bash
# 一次性重建所有服务
docker-compose up -d --build
```

## 💡 开发提示

### 开发环境配置
为了避免每次修改都要重新构建，建议在开发时：

1. **使用本地开发服务器**
```bash
cd frontend
npm install
npm run dev
```
访问 http://localhost:5173（Vite开发服务器）

2. **配置代理**
在 `vite.config.js` 中：
```javascript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

### 生产环境部署
```bash
# 构建优化的生产版本
docker-compose build frontend --no-cache
docker-compose up -d frontend
```

## 🎯 本次更新验证清单

- [ ] 前端镜像已重新构建（无缓存）
- [ ] 前端容器已重启
- [ ] 浏览器缓存已清理（Ctrl+Shift+R）
- [ ] 分析页面显示"数据来源"选择器
- [ ] 分析页面显示"选择数据集"下拉菜单
- [ ] 数据集列表自动加载（打开控制台查看API调用）
- [ ] 选择数据集后分析功能正常工作

## 📞 故障排除

### 问题: 更新仍未生效
1. 确认Docker镜像ID已变化：
```bash
docker images | grep frontend
```

2. 确认容器使用的是新镜像：
```bash
docker-compose ps
docker inspect mine_platform_frontend | grep Image
```

3. 强制清理并重建：
```bash
docker-compose down
docker system prune -f
docker-compose up -d --build
```

### 问题: 浏览器显示旧页面
- 清理浏览器缓存和Cookie
- 使用无痕模式访问
- 检查Service Worker（F12 → Application → Service Workers → Unregister）

