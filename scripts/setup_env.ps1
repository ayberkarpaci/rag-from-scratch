# Ortam kurulumu. Proje kokunden calistirilir.

$root = Join-Path $PSScriptRoot ".."
$venv = Join-Path $root ".venv"
$requirements = Join-Path $root "requirements.txt"

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $venv)) {
    Write-Host "venv olusturuluyor..."
    python -m venv $venv
}

& "$venv\Scripts\Activate.ps1"
pip install -q -r $requirements

Write-Host "Ortam hazir. Aktive etmek icin:"
Write-Host "  .venv\Scripts\Activate.ps1"
