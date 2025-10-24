@echo off
REM ============================================
REM 集成矿山智能预测平台 - Docker 一键部署脚本 (Windows)
REM ============================================

echo ========================================
echo 集成矿山智能预测平台 - Docker 部署
echo ========================================
echo.

REM 检查 Docker 是否安装
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到 Docker，请先安装 Docker Desktop
    echo 下载地址: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM 检查 Docker 是否运行
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker 未运行，请启动 Docker Desktop
    pause
    exit /b 1
)

echo [信息] Docker 环境检查通过
echo.

REM 检查 .env 文件
if not exist .env (
    echo [警告] 未找到 .env 文件，正在从 .env.example 复制...
    copy .env.example .env
    echo [提示] 请编辑 .env 文件配置您的环境变量
    echo.
)

REM 创建必要的目录
echo [信息] 创建必要的目录...
if not exist media mkdir media
if not exist logs mkdir logs
if not exist staticfiles mkdir staticfiles
if not exist nginx\ssl mkdir nginx\ssl

echo.
echo 请选择操作:
echo 1. 首次部署（构建并启动）
echo 2. 启动服务
echo 3. 停止服务
echo 4. 重启服务
echo 5. 查看日志
echo 6. 清理并重新部署
echo 7. 停止并删除所有容器、卷
echo 0. 退出
echo.

set /p choice=请输入选项 (0-7): 

if "%choice%"=="1" goto first_deploy
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto restart
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto rebuild
if "%choice%"=="7" goto cleanup
if "%choice%"=="0" goto end
goto end

:first_deploy
echo.
echo [信息] 开始首次部署...
echo [信息] 构建 Docker 镜像...
docker-compose build
echo [信息] 启动服务...
docker-compose up -d
echo [信息] 等待服务启动...
timeout /t 10 /nobreak >nul
echo.
echo [成功] 部署完成！
echo.
echo 访问地址:
echo   - 前端: http://localhost
echo   - 后端 API: http://localhost:8000
echo   - 后端管理: http://localhost:8000/admin
echo   - 默认管理员: admin / admin123456
echo.
goto end

:start
echo.
echo [信息] 启动服务...
docker-compose up -d
echo [成功] 服务已启动
goto end

:stop
echo.
echo [信息] 停止服务...
docker-compose down
echo [成功] 服务已停止
goto end

:restart
echo.
echo [信息] 重启服务...
docker-compose restart
echo [成功] 服务已重启
goto end

:logs
echo.
echo [信息] 查看实时日志 (按 Ctrl+C 退出)...
docker-compose logs -f
goto end

:rebuild
echo.
echo [警告] 这将停止服务并重新构建所有镜像
set /p confirm=确认继续? (Y/N): 
if /i not "%confirm%"=="Y" goto end
echo [信息] 停止服务...
docker-compose down
echo [信息] 清理旧镜像...
docker-compose build --no-cache
echo [信息] 启动服务...
docker-compose up -d
echo [成功] 重新部署完成
goto end

:cleanup
echo.
echo [警告] 这将删除所有容器、网络、卷和镜像！数据将丢失！
set /p confirm=确认继续? (Y/N): 
if /i not "%confirm%"=="Y" goto end
echo [信息] 停止并删除所有容器...
docker-compose down -v
echo [信息] 删除镜像...
docker-compose down --rmi all
echo [成功] 清理完成
goto end

:end
echo.
echo 按任意键退出...
pause >nul
