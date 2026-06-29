# study_rag justfile
# 安装 just: https://github.com/casey/just
#   winget install casey.just
#   choco install just
#
# 列出所有命令:  just
# 运行命令:      just <recipe>

# ---- 平台配置: Windows 上显式使用 PowerShell ----
# (just 默认找 sh，Windows 上没有；显式指定 powershell 才能跑 pip / npm 等命令)
# Linux/macOS 部署时把这一行注释掉
set shell := ["powershell", "-NoLogo", "-Command"]
set windows-shell := ["powershell", "-NoLogo", "-Command"]

# ---- 变量 ----
# 默认使用激活 venv 里的 python；如果没激活则用系统 python。
# 激活 venv 后 `python` 自动指向 venv interpreter，justfile 无需改。
# 显式覆盖: just py=python3.11 admin
py := "venv/Scripts/python.exe"
pytest_args := "-v"
admin_port := "3200"
mcp_port := "3220"

# ---- 命令列表 ----

# 显示可用命令
default:
    @just --list

# ---- 虚拟环境 ----
# 项目 venv 在 .venv/，已在 .gitignore 里。
# 一次性初始化：创建 venv + 安装核心依赖 + dev/llamaindex 扩展包
# 已存在则跳过创建；幂等可重复运行

# 初始化项目：创建 venv + 安装核心 + 开发依赖
setup:
    if (-not (Test-Path .venv)) { python -m venv .venv }
    @echo "✅ venv 已就绪: .venv"
    @echo "👉 激活方式: .\.venv\Scripts\Activate.ps1   (PowerShell)"
    @echo "           source .venv/bin/activate       (bash/zsh)"
    {{py}} -m pip install -U pip
    {{py}} -m pip install -e ".[dev,llamaindex]"

# 显示当前使用的 python / venv 信息
venv-info:
    @{{py}} -c "import sys, os; venv=os.environ.get('VIRTUAL_ENV', '<未激活>'); print('python :', sys.executable); print('版本   :', sys.version.split()[0]); print('venv   :', venv)"

# 删除项目虚拟环境
clean-venv:
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

# 安装可选依赖组到当前 python
# 用法: just install-extra vector-milvus
#       just install-extra vector-milvus,embedding-openai
install-extra GROUP:
    {{py}} -m pip install -e ".[{{GROUP}}]"

# ---- 安装 / 开发工作流 ----

# 安装项目 + 开发依赖（venv 不存在则自动创建；幂等）
install:
    if (-not (Test-Path .venv)) { python -m venv .venv }
    {{py}} -m pip install -U pip
    {{py}} -m pip install -e ".[dev,llamaindex]"

# 运行全部测试
test:
    {{py}} -m pytest tests/ {{pytest_args}}

# 运行单个测试文件
test-one FILE:
    {{py}} -m pytest tests/{{FILE}} {{pytest_args}}

# 使用 ruff 进行代码检查
lint:
    {{py}} -m ruff check src/ tests/

# 自动修复代码检查问题
lint-fix:
    {{py}} -m ruff check --fix src/ tests/

# 使用 mypy 进行类型检查
typecheck:
    {{py}} -m mypy src/

# 全部质量门禁（代码检查 + 类型检查 + 测试）
verify: lint typecheck test

# 启动 admin REST 服务（端口 3200）
admin:
    {{py}} -m uvicorn study_rag.app:app --host 0.0.0.0 --port {{admin_port}} --reload

# 启动 MCP 独立服务（端口 3220，streamable_http）
mcp:
    {{py}} -m uvicorn study_rag.mcp_standalone:app --host 0.0.0.0 --port {{mcp_port}} --reload

# 启动 MCP Inspector UI（在浏览器打开 localhost:6274，已预配置指向我们的 MCP 服务）
# 需要: 在另一个终端先运行 `just mcp`
#
# 实现：把 PowerShell 多行脚本放到 scripts/inspector.ps1
#   - just recipe body 不支持多行缩进（会报 "extra leading whitespace"）
#   - 单独 .ps1 文件避免 just+PowerShell $ 变量冲突（just $$  → PS $ 的双重转义）
#
# 工作流程见 scripts/inspector.ps1 注释
inspector:
    powershell -NoLogo -ExecutionPolicy Bypass -File scripts/inspector.ps1 -McpPort {{mcp_port}}

# MCP Inspector CLI 模式：列出所有工具
# 需要: 在另一个终端先运行 `just mcp`
inspector-tools:
    @echo "从 http://localhost:{{mcp_port}}/mcp 列出 MCP 工具 ..."
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method tools/list

# MCP Inspector CLI 模式：列出所有资源
inspector-resources:
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method resources/list

# MCP Inspector CLI 模式：列出所有提示词
inspector-prompts:
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method prompts/list

# 在两个终端分别启动 admin + mcp
dev:
    @echo "在一个终端运行 'just admin'，另一个终端运行 'just mcp'"

# ---- Docker ----

# 构建 Docker 镜像（EXTRAS 控制可选依赖组，默认 llamaindex,vector-milvus）
docker-build EXTRAS="llamaindex,vector-milvus":
    docker build -f docker/Dockerfile --build-arg EXTRAS={{EXTRAS}} -t study-rag:dev .

# 通过 docker compose 启动 admin + mcp（自动从项目根目录 .env 读取环境变量）
docker-up:
    docker compose -f docker/docker-compose.yml up -d --build

# 通过 docker compose 启动 admin + mcp + Milvus
docker-up-vector:
    docker compose -f docker/docker-compose.yml --profile vector up -d --build

# 跟踪所有服务的日志
docker-logs:
    docker compose -f docker/docker-compose.yml logs -f

# 停止 docker compose（保留数据卷）
docker-down:
    docker compose -f docker/docker-compose.yml down

# 停止 docker compose 并删除数据卷（Milvus / rag-data）
docker-purge:
    docker compose -f docker/docker-compose.yml down -v

# 在 Docker 容器内运行测试
docker-test:
    docker compose -f docker/docker-compose.yml --profile dev run --rm dev-tools just test

# ---- 快捷命令 ----

# 打开已加载项目的 REPL
repl:
    {{py}} -c "from study_rag import *; import IPython; IPython.start_ipython()"

# ---- admin UI (React + Vite + antd) ----
# 注：Windows PowerShell 5.x 不支持 `&&`；用 `; if ($?)` 等价
# PowerShell 7+ 改成 `&&` 也行，但兼容性优先

# 安装前端依赖
ui-install:
    Set-Location frontend; if ($?) { npm install }

# 启动 Vite 开发服务器（带 HMR，代理 /admin /metrics /mcp 到 FastAPI）
ui-dev:
    Set-Location frontend; if ($?) { npm run dev }

# 构建 SPA 到 src/study_rag/web/dist（由 FastAPI 在 /admin/ui/ 提供服务）
ui-build:
    Set-Location frontend; if ($?) { npm run build }

# 前端类型检查（不输出文件）
ui-typecheck:
    Set-Location frontend; if ($?) { npm run typecheck }

# 清理构建产物
ui-clean:
    if (Test-Path frontend/node_modules) { Remove-Item -Recurse -Force frontend/node_modules }
    if (Test-Path src/study_rag/web/dist) { Remove-Item -Recurse -Force src/study_rag/web/dist }

# 全部质量门禁（含前端：代码检查 + 类型检查 + 测试 + 前端类型检查）
verify-all: lint typecheck test ui-typecheck