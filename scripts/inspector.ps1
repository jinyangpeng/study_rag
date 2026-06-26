# MCP Inspector 启动器
# 用法：pwsh scripts/inspector.ps1 [mcp_port]
#
# 工作流程：
#   1. 设置 DANGEROUSLY_OMIT_AUTH=true 禁用 proxy 鉴权
#   2. 启动 npx @modelcontextprotocol/inspector（proxy + UI）
#   3. 等待 UI 就绪（port 6274），自动开浏览器
#
# 为什么不抓 token 拼 URL：
#   - MCP Inspector v0.15+ UI 不会从 URL 读 MCP_PROXY_AUTH_TOKEN
#     （config.MCP_PROXY_AUTH_TOKEN.value 默认为空，没有 searchParam 解析）
#   - 用户必须手动在 UI Configuration 里填 token，体感很差
#   - 官方逃生口 DANGEROUSLY_OMIT_AUTH=true 在 PowerShell + npx 下稳定可用
#
# 为什么不直接 Start-Process npx：
#   - 旧方案抓 token 拼 URL 在 v0.15 已失效（UI 不读 URL）
#   - 改用「禁用鉴权 + 等待 UI 就绪 + 自动开浏览器」最稳
#
# 实现注意：
#   - 用 Get-Content -Raw 代替 [System.IO.File]::ReadAllText
#   - 脚本文件存为 UTF-8 with BOM

[CmdletBinding()]
param(
    [int]$McpPort = 8001
)

# ---- helper ----
function Read-FileOrEmpty {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    try {
        return (Get-Content -LiteralPath $Path -Raw -ErrorAction Stop)
    } catch {
        return ""
    }
}

# ---- start ----
Write-Host "Starting MCP Inspector (UI mode)..." -ForegroundColor Cyan
Write-Host "  - MCP Server: http://localhost:${McpPort}/mcp (streamable-http)" -ForegroundColor Yellow
Write-Host "  - DANGEROUSLY_OMIT_AUTH=true (proxy auth disabled for local dev)" -ForegroundColor Yellow
Write-Host ""

# ---- 修复 console 乱码 ----
# npx 子进程 stdout 是 UTF-8（含 emoji），PowerShell 5.x 默认 OEM 代码页（GBK/cp936）
# 必须把 console 切到 UTF-8，否则 emoji 解码失败显示成 "鈿欙笍" 这种乱码
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 > $null
} catch {
    # 某些 console 不支持改编码，忽略
}

# ---- 清理可能存在的僵尸 inspector 进程 ----
# 之前 Start-Process 启动的 npx 子进程在 PowerShell 退出后可能没被清理，
# 旧 proxy（鉴权开启版本）仍占着 6277 端口，导致新 proxy 启动后浏览器仍命中旧进程 → 401。
$portsToClean = @(6274, 6277)
foreach ($port in $portsToClean) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        $pidToKill = $conn.OwningProcess
        try {
            $procInfo = Get-Process -Id $pidToKill -ErrorAction Stop
            Write-Host ("Killing stale process on port {0}: PID {1} ({2})" -f $port, $pidToKill, $procInfo.ProcessName) -ForegroundColor Yellow
            Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
        } catch {
            # 进程已不存在
        }
    }
}
Start-Sleep -Milliseconds 500  # 等端口释放

# 设置 env（子进程 npx 会继承）
$env:DANGEROUSLY_OMIT_AUTH = "true"
# Node.js 默认 UTF-8，但显式设置 PYTHONIOENCODING 防御性
$env:PYTHONIOENCODING = "utf-8"

# 启动 inspector（后台进程，stdout/stderr 重定向到临时文件）
$outFile = Join-Path $env:TEMP "inspector-$PID.out"
$errFile = Join-Path $env:TEMP "inspector-$PID.err"

$proc = Start-Process -FilePath "npx" `
    -ArgumentList "--yes", "@modelcontextprotocol/inspector" `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $outFile `
    -RedirectStandardError $errFile

# 等 UI 起来后自动开浏览器
$timeout = 45
$elapsed = 0
$pollIntervalMs = 500
$uiReady = $false

while (-not $proc.HasExited -and -not $uiReady -and $elapsed -lt $timeout) {
    Start-Sleep -Milliseconds $pollIntervalMs
    $elapsed += ($pollIntervalMs / 1000.0)

    # 探测 port 6274 是否就绪
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect("127.0.0.1", 6274, $null, $null)
        $success = $iar.AsyncWaitHandle.WaitOne(100, $false)
        if ($success) {
            $client.EndConnect($iar)
            $uiReady = $true
        }
        $client.Close()
    } catch {
        # 端口未就绪，继续等
    }
}

# 把 inspector 启动时的所有输出一次性打印到 console
$finalContent = (Read-FileOrEmpty $outFile) + (Read-FileOrEmpty $errFile)
if ($finalContent) {
    Write-Host $finalContent
}

if ($uiReady) {
    $url = "http://localhost:6274/?transport=streamable-http&serverUrl=http://localhost:${McpPort}/mcp"
    Write-Host ""
    Write-Host "Inspector UI ready, opening browser..." -ForegroundColor Green
    Write-Host "  $url"

    # ---- 用临时 user-data-dir 启动浏览器，避免 localStorage 残留 ----
    # inspector 0.15 UI 启动时遍历 baseConfig，如果 localStorage 里有
    # 新版已删除的 key，DEFAULT_INSPECTOR_CONFIG[key] = undefined → .label 报错 → 空白页
    # 用 --user-data-dir 启动全新 profile，localStorage 干净，永不踩雷
    $tempProfile = Join-Path $env:TEMP "inspector-chrome-profile"
    $browserLaunched = $false

    # 优先用 Chrome / Edge / Brave（按 PATH 顺序找）
    $browserCandidates = @(
        "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles}\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
    )
    foreach ($browser in $browserCandidates) {
        if (Test-Path -LiteralPath $browser) {
            Write-Host ("  Launching browser: {0}" -f $browser) -ForegroundColor DarkGray
            Start-Process -FilePath $browser `
                -ArgumentList @(
                    "--user-data-dir=$tempProfile",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--app=$url"
                ) | Out-Null
            $browserLaunched = $true
            break
        }
    }

    if (-not $browserLaunched) {
        # fallback: 系统默认浏览器（可能有 localStorage 残留风险）
        Write-Host "  (Chrome/Edge not found; using system default browser)" -ForegroundColor DarkGray
        Write-Host "  If UI is blank, run in browser console:" -ForegroundColor Yellow
        Write-Host "    localStorage.clear(); sessionStorage.clear(); location.reload()" -ForegroundColor Yellow
        Start-Process $url
    }
} else {
    Write-Host ""
    Write-Host ("WARNING: UI not ready within {0}s; please open http://localhost:6274 manually" -f $timeout) -ForegroundColor Red
}

# 等 inspector 进程（用户 Ctrl+C 退出）
try {
    Wait-Process -Id $proc.Id -ErrorAction Stop
} catch {
    # 进程已退出
}

# 清理临时文件
Remove-Item -Force -ErrorAction SilentlyContinue $outFile, $errFile

