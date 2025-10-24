#!/bin/bash
# ============================================
# 集成矿山智能预测平台 - Docker 一键部署脚本 (Linux/Mac)
# ============================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "集成矿山智能预测平台 - Docker 部署"
echo -e "========================================${NC}"
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[错误] 未检测到 Docker，请先安装 Docker${NC}"
    echo "安装指南: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}[错误] 未检测到 Docker Compose，请先安装${NC}"
    echo "安装指南: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo -e "${RED}[错误] Docker 未运行，请启动 Docker 服务${NC}"
    exit 1
fi

echo -e "${GREEN}[信息] Docker 环境检查通过${NC}"
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo -e "${YELLOW}[警告] 未找到 .env 文件，正在从 .env.example 复制...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}[提示] 请编辑 .env 文件配置您的环境变量${NC}"
    echo ""
fi

# 创建必要的目录
echo -e "${GREEN}[信息] 创建必要的目录...${NC}"
mkdir -p media logs staticfiles nginx/ssl

# 显示菜单
show_menu() {
    echo ""
    echo "请选择操作:"
    echo "1. 首次部署（构建并启动）"
    echo "2. 启动服务"
    echo "3. 停止服务"
    echo "4. 重启服务"
    echo "5. 查看日志"
    echo "6. 查看服务状态"
    echo "7. 清理并重新部署"
    echo "8. 进入后端容器"
    echo "9. 数据库备份"
    echo "10. 停止并删除所有容器、卷"
    echo "0. 退出"
    echo ""
}

# 首次部署
first_deploy() {
    echo ""
    echo -e "${GREEN}[信息] 开始首次部署...${NC}"
    echo -e "${GREEN}[信息] 构建 Docker 镜像...${NC}"
    docker-compose build
    echo -e "${GREEN}[信息] 启动服务...${NC}"
    docker-compose up -d
    echo -e "${GREEN}[信息] 等待服务启动...${NC}"
    sleep 10
    echo ""
    echo -e "${GREEN}[成功] 部署完成！${NC}"
    echo ""
    echo "访问地址:"
    echo "  - 前端: http://localhost"
    echo "  - 后端 API: http://localhost:8000"
    echo "  - 后端管理: http://localhost:8000/admin"
    echo "  - 默认管理员: admin / admin123456"
    echo ""
}

# 启动服务
start_services() {
    echo ""
    echo -e "${GREEN}[信息] 启动服务...${NC}"
    docker-compose up -d
    echo -e "${GREEN}[成功] 服务已启动${NC}"
}

# 停止服务
stop_services() {
    echo ""
    echo -e "${YELLOW}[信息] 停止服务...${NC}"
    docker-compose down
    echo -e "${GREEN}[成功] 服务已停止${NC}"
}

# 重启服务
restart_services() {
    echo ""
    echo -e "${GREEN}[信息] 重启服务...${NC}"
    docker-compose restart
    echo -e "${GREEN}[成功] 服务已重启${NC}"
}

# 查看日志
view_logs() {
    echo ""
    echo -e "${GREEN}[信息] 查看实时日志 (按 Ctrl+C 退出)...${NC}"
    docker-compose logs -f
}

# 查看状态
view_status() {
    echo ""
    echo -e "${GREEN}[信息] 服务状态:${NC}"
    docker-compose ps
}

# 重新部署
rebuild() {
    echo ""
    echo -e "${YELLOW}[警告] 这将停止服务并重新构建所有镜像${NC}"
    read -p "确认继续? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "已取消"
        return
    fi
    echo -e "${GREEN}[信息] 停止服务...${NC}"
    docker-compose down
    echo -e "${GREEN}[信息] 清理旧镜像...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}[信息] 启动服务...${NC}"
    docker-compose up -d
    echo -e "${GREEN}[成功] 重新部署完成${NC}"
}

# 进入后端容器
enter_backend() {
    echo ""
    echo -e "${GREEN}[信息] 进入后端容器...${NC}"
    docker-compose exec backend /bin/bash
}

# 数据库备份
backup_database() {
    echo ""
    BACKUP_DIR="./backups"
    mkdir -p $BACKUP_DIR
    BACKUP_FILE="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"
    echo -e "${GREEN}[信息] 备份数据库到 $BACKUP_FILE${NC}"
    docker-compose exec -T db pg_dump -U postgres mine_platform_db > $BACKUP_FILE
    echo -e "${GREEN}[成功] 数据库备份完成${NC}"
}

# 清理所有
cleanup() {
    echo ""
    echo -e "${RED}[警告] 这将删除所有容器、网络、卷和镜像！数据将丢失！${NC}"
    read -p "确认继续? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "已取消"
        return
    fi
    echo -e "${GREEN}[信息] 停止并删除所有容器...${NC}"
    docker-compose down -v
    echo -e "${GREEN}[信息] 删除镜像...${NC}"
    docker-compose down --rmi all
    echo -e "${GREEN}[成功] 清理完成${NC}"
}

# 主循环
while true; do
    show_menu
    read -p "请输入选项 (0-10): " choice
    
    case $choice in
        1) first_deploy ;;
        2) start_services ;;
        3) stop_services ;;
        4) restart_services ;;
        5) view_logs ;;
        6) view_status ;;
        7) rebuild ;;
        8) enter_backend ;;
        9) backup_database ;;
        10) cleanup ;;
        0) 
            echo -e "${GREEN}再见！${NC}"
            exit 0 
            ;;
        *) 
            echo -e "${RED}无效选项，请重新选择${NC}" 
            ;;
    esac
done
