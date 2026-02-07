$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONIOENCODING = "utf-8"
$env:ASSISTANT_API_BASE = "http://127.0.0.1:4000"

$dataDir = Join-Path $root "data"
$backendDir = Join-Path $root "apps\backend"

Write-Host "Running pipeline using $dataDir ..."
py scripts\run_full_pipeline.py --data-dir $dataDir --output-dir reports

Write-Host "Starting API server on http://127.0.0.1:5000/api/health ..."
Start-Process -FilePath "py" -ArgumentList "scripts\run_api.py" -WorkingDirectory $root

if (Test-Path $backendDir) {
    if (-not (Test-Path (Join-Path $backendDir "node_modules"))) {
        Write-Host "Installing assistant backend dependencies ..."
        Start-Process -FilePath "npm" -ArgumentList "install" -WorkingDirectory $backendDir -Wait
    }

    Write-Host "Starting assistant API on http://127.0.0.1:4000/api/health ..."
    Start-Process -FilePath "npm" -ArgumentList "run dev" -WorkingDirectory $backendDir
}

Write-Host "Starting Streamlit UI on http://127.0.0.1:8501/ ..."
Start-Process -FilePath "py" -ArgumentList "-m streamlit run app_streamlit.py --server.port 8501 --server.address 127.0.0.1" -WorkingDirectory $root

Write-Host "All services started."
