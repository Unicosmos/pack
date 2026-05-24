#!/usr/bin/env bash
# ============================================================
# Pack Web 集成启动脚本 - Linux/macOS
# 同时启动前后端服务，提供统一管理界面
# ============================================================
#
# 功能：
# 1. 自动检测并激活Python虚拟环境
# 2. 启动FastAPI后端服务（端口8000）
# 3. 启动Vite前端开发服务器（端口5173）
# 4. 提供日志输出和状态反馈
# 5. 支持Ctrl+C统一停止所有服务
#
# 用法：
#   ./start.sh              # 启动所有服务
#   ./start.sh --no-install # 跳过依赖安装
#   ./start.sh --skip-frontend  # 仅启动后端
#   ./start.sh --skip-backend  # 仅启动前端
#   ./start.sh --clean      # 清理后重新启动
#
# 要求：
#   - Python 3.9+
#   - Node.js 18+
#   - npm 或 yarn
#
# ============================================================

set -e

# 配置区域
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
BACKEND_DIR="$PROJECT_ROOT/web/backend"
FRONTEND_DIR="$PROJECT_ROOT/web/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_URL="http://localhost:$BACKEND_PORT"
FRONTEND_URL="http://localhost:$FRONTEND_PORT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 全局变量
SKIP_INSTALL=false
SKIP_FRONTEND=false
SKIP_BACKEND=false
CLEAN=false
BACKEND_PID=""
FRONTEND_PID=""

# ============================================================
# 函数定义
# ============================================================

log_info() {
    echo -e "${NC}[${BLUE}INFO${NC}] $1"
}

log_success() {
    echo -e "${NC}[${GREEN}SUCCESS${NC}] $1"
}

log_warning() {
    echo -e "${NC}[${YELLOW}WARNING${NC}] $1"
}

log_error() {
    echo -e "${NC}[${RED}ERROR${NC}] $1"
}

log_backend() {
    echo -e "${NC}[${MAGENTA}BACKEND${NC}] $1"
}

log_frontend() {
    echo -e "${NC}[${BLUE}FRONTEND${NC}] $1"
}

print_banner() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Pack Web 集成启动脚本 v1.0.0${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

get_pid_by_port() {
    local port=$1
    if [[ "$OSTYPE" == "darwin"* ]]; then
        lsof -ti :$port 2>/dev/null || true
    else
        lsof -ti :$port 2>/dev/null || true
    fi
}

stop_service_by_port() {
    local port=$1
    local name=$2
    local pid=$(get_pid_by_port $port)

    if [[ -n "$pid" ]]; then
        log_warning "[$name] 正在停止端口 $port 上的进程 (PID: $pid)..."
        kill -TERM $pid 2>/dev/null || true
        sleep 1
        kill -9 $pid 2>/dev/null || true
        log_success "[$name] 已停止"
    fi
}

init_venv() {
    log_info "[虚拟环境] 检查虚拟环境..."

    if [[ ! -d "$VENV_PATH" ]]; then
        log_warning "[虚拟环境] 未找到虚拟环境，正在创建..."
        cd "$PROJECT_ROOT"
        python3 -m venv .venv
        log_success "[虚拟环境] 虚拟环境创建成功"
    fi

    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PYTHON="$VENV_PATH/bin/python"
    else
        PYTHON="$VENV_PATH/Scripts/python.exe"
    fi

    if [[ ! -f "$PYTHON" ]]; then
        log_error "[虚拟环境] 虚拟环境Python不存在，重新创建..."
        rm -rf "$VENV_PATH"
        init_venv
        return
    fi

    log_success "[虚拟环境] 已激活 (.venv)"
    echo "$PYTHON"
}

install_deps() {
    local python=$1

    if [[ "$SKIP_INSTALL" == true ]]; then
        log_warning "[依赖] 跳过安装步骤"
        return
    fi

    log_info "[依赖] 检查Python依赖..."

    local req_file="$PROJECT_ROOT/requirements.txt"
    if [[ ! -f "$req_file" ]]; then
        log_warning "[依赖] 未找到requirements.txt，跳过"
        return
    fi

    log_info "[依赖] 安装Python依赖（如果需要）..."
    "$python" -m pip install --upgrade pip --quiet 2>/dev/null || true
    "$python" -m pip install -r "$req_file" --quiet 2>/dev/null || true
    log_success "[依赖] Python依赖检查完成"

    log_info "[依赖] 检查Node.js依赖..."
    if check_command npm; then
        cd "$FRONTEND_DIR"
        if [[ ! -d "node_modules" ]]; then
            log_info "[依赖] 安装前端依赖..."
            npm install --silent
        else
            log_success "[依赖] 前端依赖已安装"
        fi
        cd "$PROJECT_ROOT"
    else
        log_error "[依赖] 未找到npm，请确保Node.js已安装"
    fi
}

start_backend() {
    local python=$1

    if [[ "$SKIP_BACKEND" == true ]]; then
        log_warning "[后端] 已跳过"
        return ""
    fi

    log_backend "[后端] 检查端口 $BACKEND_PORT 是否被占用..."
    stop_service_by_port $BACKEND_PORT "Backend"

    local backend_script="$BACKEND_DIR/main.py"
    if [[ ! -f "$backend_script" ]]; then
        log_error "[后端] 未找到main.py: $backend_script"
        return ""
    fi

    log_backend "[后端] 启动FastAPI服务 (http://0.0.0.0:$BACKEND_PORT)..."

    local log_dir="$PROJECT_ROOT/logs"
    local log_file="$log_dir/backend.log"
    local error_file="$log_dir/backend_error.log"

    mkdir -p "$log_dir"

    cd "$BACKEND_DIR"
    "$python" -m uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > "$log_file" 2> "$error_file" &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"

    sleep 3

    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log_error "[后端] 启动失败，请检查日志: $log_file"
        log_error "[后端] 错误信息: $(tail -10 $error_file 2>/dev/null || echo '无')"
        return ""
    fi

    log_success "[后端] 已启动 (PID: $BACKEND_PID)"
    log_backend "[后端] API文档: $BACKEND_URL/docs"

    echo $BACKEND_PID
}

start_frontend() {
    if [[ "$SKIP_FRONTEND" == true ]]; then
        log_warning "[前端] 已跳过"
        return ""
    fi

    log_frontend "[前端] 检查端口 $FRONTEND_PORT 是否被占用..."
    stop_service_by_port $FRONTEND_PORT "Frontend"

    if ! check_command npm; then
        log_error "[前端] 未找到npm命令，请安装Node.js"
        return ""
    fi

    log_frontend "[前端] 启动Vite开发服务器 (http://localhost:$FRONTEND_PORT)..."

    local log_dir="$PROJECT_ROOT/logs"
    local log_file="$log_dir/frontend.log"
    local error_file="$log_dir/frontend_error.log"

    mkdir -p "$log_dir"

    cd "$FRONTEND_DIR"
    npm run dev -- --port $FRONTEND_PORT --host > "$log_file" 2> "$error_file" &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"

    sleep 5

    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        log_error "[前端] 启动失败，请检查日志: $log_file"
        return ""
    fi

    log_success "[前端] 已启动 (PID: $FRONTEND_PID)"
    log_frontend "[前端] 访问地址: $FRONTEND_URL"

    echo $FRONTEND_PID
}

cleanup() {
    echo ""
    log_warning "[停止] 正在停止所有服务..."

    if [[ -n "$BACKEND_PID" ]] && kill -0 $BACKEND_PID 2>/dev/null; then
        log_backend "[后端] 停止中 (PID: $BACKEND_PID)..."
        kill -TERM $BACKEND_PID 2>/dev/null || true
        sleep 1
        kill -9 $BACKEND_PID 2>/dev/null || true
    fi
    stop_service_by_port $BACKEND_PORT "Backend"

    if [[ -n "$FRONTEND_PID" ]] && kill -0 $FRONTEND_PID 2>/dev/null; then
        log_frontend "[前端] 停止中 (PID: $FRONTEND_PID)..."
        kill -TERM $FRONTEND_PID 2>/dev/null || true
        sleep 1
        kill -9 $FRONTEND_PID 2>/dev/null || true
    fi
    stop_service_by_port $FRONTEND_PORT "Frontend"

    log_success "[停止] 所有服务已停止"
}

wait_services() {
    local backend_pid=$1
    local frontend_pid=$2

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Pack Web 服务运行中${NC}"
    echo -e "${CYAN}========================================${NC}"

    local services=()
    [[ -n "$backend_pid" ]] && services+=("后端 API (端口 $BACKEND_PORT)")
    [[ -n "$frontend_pid" ]] && services+=("前端 UI (端口 $FRONTEND_PORT)")

    echo -e "${GREEN}运行服务: ${services[*]}${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"
    echo ""

    local exit_count=0

    while true; do
        sleep 2

        if [[ -n "$backend_pid" ]] && ! kill -0 $backend_pid 2>/dev/null; then
            log_error "[警告] 后端服务已意外退出"
            exit_count=$((exit_count + 1))
        fi

        if [[ -n "$frontend_pid" ]] && ! kill -0 $frontend_pid 2>/dev/null; then
            log_error "[警告] 前端服务已意外退出"
            exit_count=$((exit_count + 1))
        fi

        if [[ $exit_count -ge 3 ]]; then
            log_error "[错误] 服务连续退出，停止监控"
            break
        fi
    done
}

show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --no-install      跳过依赖安装步骤"
    echo "  --skip-frontend   仅启动后端服务"
    echo "  --skip-backend    仅启动前端服务"
    echo "  --clean           清理之前的服务后重新启动"
    echo "  -h, --help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 启动所有服务"
    echo "  $0 --no-install       # 跳过安装，快速启动"
    echo "  $0 --skip-frontend    # 仅启动后端"
}

# ============================================================
# 参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --skip-backend)
            SKIP_BACKEND=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# 主流程
# ============================================================

trap cleanup EXIT INT TERM

print_banner

if [[ "$CLEAN" == true ]]; then
    log_warning "[清理] 正在清理之前的服务..."
    stop_service_by_port $BACKEND_PORT "Backend"
    stop_service_by_port $FRONTEND_PORT "Frontend"
    log_success "[清理] 完成"
fi

PYTHON=$(init_venv)
install_deps "$PYTHON"

BACKEND_PID=$(start_backend "$PYTHON")
FRONTEND_PID=$(start_frontend)

if [[ "$SKIP_BACKEND" == false ]] && [[ -z "$BACKEND_PID" ]]; then
    log_error "后端服务启动失败"
    exit 1
fi

if [[ "$SKIP_FRONTEND" == false ]] && [[ -z "$FRONTEND_PID" ]]; then
    log_error "前端服务启动失败"
    exit 1
fi

wait_services "$BACKEND_PID" "$FRONTEND_PID"
