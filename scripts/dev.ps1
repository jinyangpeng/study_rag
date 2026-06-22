# 开发脚本
# 用法：powershell -File scripts/dev.ps1 <command>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("admin", "mcp", "inspector", "test", "lint", "typecheck", "all-verify", "help")]
    [string]$Command
)

switch ($Command) {
    "admin" {
        Write-Host ">>> 启动 admin REST API (port 8765)" -ForegroundColor Cyan
        python -m uvicorn study_rag.app:app --host 127.0.0.1 --port 8765 --reload
    }
    "mcp" {
        Write-Host ">>> 启动 MCP standalone server (port 8001, streamable_http)" -ForegroundColor Cyan
        $env:MCP_PORT = "8001"
        python -m uvicorn study_rag.mcp_standalone:app --host 127.0.0.1 --port 8001 --reload
    }
    "inspector" {
        Write-Host ">>> 启动 MCP Inspector (需先启动 mcp)" -ForegroundColor Cyan
        Write-Host "    Inspector UI: http://localhost:5173" -ForegroundColor Yellow
        Write-Host "    MCP endpoint: http://localhost:8001/mcp" -ForegroundColor Yellow
        npx --yes @modelcontextprotocol/inspector `
            --transport streamable-http `
            --server-url http://localhost:8001/mcp
    }
    "test" {
        Write-Host ">>> 跑 pytest + 所有 verify 脚本" -ForegroundColor Cyan
        Write-Host "`n--- pytest tests/ ---" -ForegroundColor Yellow
        python -m pytest tests/ -v
        if ($LASTEXITCODE -ne 0) { exit 1 }
        $scripts = Get-ChildItem -Path tests -Filter "verify_*.py" | Sort-Object Name
        foreach ($s in $scripts) {
            Write-Host "`n--- tests/$($s.Name) ---" -ForegroundColor Yellow
            python "tests/$($s.Name)"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAIL: tests/$($s.Name)" -ForegroundColor Red
                exit 1
            }
        }
    }
    "lint" {
        Write-Host ">>> ruff check" -ForegroundColor Cyan
        python -m ruff check src/ tests/
    }
    "typecheck" {
        Write-Host ">>> mypy" -ForegroundColor Cyan
        python -m mypy src/
    }
    "all-verify" {
        & $PSCommandPath -Command lint
        if ($LASTEXITCODE -ne 0) { exit 1 }
        & $PSCommandPath -Command typecheck
        if ($LASTEXITCODE -ne 0) { exit 1 }
        & $PSCommandPath -Command test
    }
    "help" {
        Write-Host @"

study_rag 开发脚本

命令:
  admin        启动 admin REST API (port 8765)
  mcp          启动 MCP streamable_http server (port 8001)
  inspector    启动 MCP Inspector（需先启动 mcp）
  test         跑 pytest + tests/verify_*.py
  lint         ruff check
  typecheck    mypy
  all-verify   lint + typecheck + test
  help         显示本帮助

推荐开发流程:
  Terminal 1: pwsh scripts/dev.ps1 mcp
  Terminal 2: pwsh scripts/dev.ps1 inspector
  Terminal 3: pwsh scripts/dev.ps1 admin (用于调试 admin REST)

MCP Inspector 调试:
  1. 启动 mcp:   pwsh scripts/dev.ps1 mcp
  2. 启动 inspector: pwsh scripts/dev.ps1 inspector
  3. 浏览器打开 http://localhost:5173
  4. 选择 transport: streamable_http
  5. 输入 URL: http://localhost:8001/mcp
  6. 点 Connect -> List Tools -> Call Tool

"@
    }
}
