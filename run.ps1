param(
    [string]$Target = "desktop"
)

if ($Target -in @("web", "--web", "serve", "server")) {
    Write-Host "Starting Swim Analyzer Web App on http://localhost:8000 ..." -ForegroundColor Cyan
    python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
} elseif ($Target -eq "main.py") {
    python main.py
} else {
    python main.py
}
