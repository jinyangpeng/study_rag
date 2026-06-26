# study_rag justfile
# Install just: https://github.com/casey/just
#   winget install casey.just
#   choco install just
#
# List all recipes:  just
# Run:               just <recipe>

# ---- platform: explicitly use PowerShell on Windows ----
# (just 默认找 sh，Windows 上没有；显式指定 powershell 才能跑 pip / npm 等命令)
# Linux/macOS 部署时把这一行注释掉
set shell := ["powershell", "-NoLogo", "-Command"]
set windows-shell := ["powershell", "-NoLogo", "-Command"]

# ---- variables ----
# 默认使用激活 venv 里的 python；如果没激活则用系统 python。
# 激活 venv 后 `python` 自动指向 venv interpreter，justfile 无需改。
# 显式覆盖: just py=python3.11 admin
py := "python"
pytest_args := "-v"
admin_port := "8765"
mcp_port := "8001"

# ---- recipes ----

# Show available recipes
default:
    @just --list

# ---- venv ----
# 项目 venv 在 .venv/，已在 .gitignore 里。
# 一次性 bootstrap：创建 venv + 装核心 + dev/llamaindex extras
# 已存在则跳过创建；幂等可重复跑

# Bootstrap project: create venv + install core + dev deps
setup:
    if (-not (Test-Path .venv)) { python -m venv .venv }
    @echo "✅ venv ready: .venv"
    @echo "👉 激活: .\.venv\Scripts\Activate.ps1   (PowerShell)"
    @echo "        source .venv/bin/activate       (bash/zsh)"
    {{py}} -m pip install -U pip
    {{py}} -m pip install -e ".[dev,llamaindex]"

# Show which python / venv is in use
venv-info:
    @{{py}} -c "import sys, os; venv=os.environ.get('VIRTUAL_ENV', '<not activated>'); print('python :', sys.executable); print('ver    :', sys.version.split()[0]); print('venv   :', venv)"

# Remove the project venv
clean-venv:
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

# Install an optional dependency group into current python
# 用法: just install-extra vector-milvus
#       just install-extra vector-milvus,embedding-openai
install-extra GROUP:
    {{py}} -m pip install -e ".[{{GROUP}}]"

# ---- install / dev workflow ----

# Install project + dev deps (creates venv if missing; idempotent)
install:
    if (-not (Test-Path .venv)) { python -m venv .venv }
    {{py}} -m pip install -U pip
    {{py}} -m pip install -e ".[dev,llamaindex]"

# Run all tests
test:
    {{py}} -m pytest tests/ {{pytest_args}}

# Run a single test file
test-one FILE:
    {{py}} -m pytest tests/{{FILE}} {{pytest_args}}

# Lint with ruff
lint:
    {{py}} -m ruff check src/ tests/

# Auto-fix lint issues
lint-fix:
    {{py}} -m ruff check --fix src/ tests/

# Type check with mypy
typecheck:
    {{py}} -m mypy src/

# All quality gates (lint + typecheck + test)
verify: lint typecheck test

# Start admin REST server (port 8765)
admin:
    {{py}} -m uvicorn study_rag.app:app --host 127.0.0.1 --port {{admin_port}} --reload

# Start MCP standalone server (port 8001, streamable_http)
mcp:
    {{py}} -m uvicorn study_rag.mcp_standalone:app --host 127.0.0.1 --port {{mcp_port}} --reload

# Start MCP Inspector UI (opens browser at localhost:6274, pre-configured for our MCP server)
# Requires: `just mcp` running in another terminal
#
# 实现：把 PowerShell 多行脚本放到 scripts/inspector.ps1
#   - just recipe body 不支持多行缩进（会报 "extra leading whitespace"）
#   - 单独 .ps1 文件避免 just+PowerShell $ 变量冲突（just $$  → PS $ 的双重转义）
#
# 工作流程见 scripts/inspector.ps1 注释
inspector:
    powershell -NoLogo -ExecutionPolicy Bypass -File scripts/inspector.ps1 -McpPort {{mcp_port}}

# MCP Inspector CLI mode: list all tools from our MCP server
# Requires: `just mcp` running in another terminal
inspector-tools:
    @echo "Listing MCP tools from http://localhost:{{mcp_port}}/mcp ..."
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method tools/list

# MCP Inspector CLI mode: list all resources
inspector-resources:
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method resources/list

# MCP Inspector CLI mode: list all prompts
inspector-prompts:
    npx --yes @modelcontextprotocol/inspector --cli http://localhost:{{mcp_port}}/mcp --transport http --method prompts/list

# Start admin + mcp in two terminals
dev:
    @echo "Run 'just admin' in one terminal and 'just mcp' in another"

# ---- docker ----

# Build Docker image (EXTRAS 控制可选依赖组，默认 llamaindex,vector-milvus)
docker-build EXTRAS="llamaindex,vector-milvus":
    docker build -f docker/Dockerfile --build-arg EXTRAS={{EXTRAS}} -t study-rag:dev .

# Start admin + mcp via docker compose（自动从项目根 .env 读取环境变量）
docker-up:
    docker compose -f docker/docker-compose.yml up -d --build

# Start admin + mcp + Milvus via docker compose
docker-up-vector:
    docker compose -f docker/docker-compose.yml --profile vector up -d --build

# Follow logs of all services
docker-logs:
    docker compose -f docker/docker-compose.yml logs -f

# Stop docker compose (keep volumes)
docker-down:
    docker compose -f docker/docker-compose.yml down

# Stop docker compose AND delete data volumes (Milvus / rag-data)
docker-purge:
    docker compose -f docker/docker-compose.yml down -v

# Run tests inside docker container
docker-test:
    docker compose -f docker/docker-compose.yml --profile dev run --rm dev-tools just test

# ---- shortcuts ----

# Open REPL with project loaded
repl:
    {{py}} -c "from study_rag import *; import IPython; IPython.start_ipython()"

# ---- admin ui (React + Vite + antd) ----
# 注：Windows PowerShell 5.x 不支持 `&&`；用 `; if ($?)` 等价
# PowerShell 7+ 改成 `&&` 也行，但兼容性优先

# Install frontend dependencies
ui-install:
    Set-Location frontend; if ($?) { npm install }

# Start Vite dev server (with HMR, proxies /admin /metrics /mcp to FastAPI)
ui-dev:
    Set-Location frontend; if ($?) { npm run dev }

# Build the SPA into src/study_rag/web/dist (served by FastAPI at /admin/ui/)
ui-build:
    Set-Location frontend; if ($?) { npm run build }

# Type-check the frontend (no emit)
ui-typecheck:
    Set-Location frontend; if ($?) { npm run typecheck }

# Clean build artifacts
ui-clean:
    if (Test-Path frontend/node_modules) { Remove-Item -Recurse -Force frontend/node_modules }
    if (Test-Path src/study_rag/web/dist) { Remove-Item -Recurse -Force src/study_rag/web/dist }

# All quality gates incl. frontend (lint + typecheck + test + ui-typecheck)
verify-all: lint typecheck test ui-typecheck
