# Environment setup. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$root = Join-Path $PSScriptRoot ".."
$venv = Join-Path $root ".venv"
$requirements = Join-Path $root "requirements.txt"

if (-not (Test-Path $venv)) {
    Write-Host "Creating venv..."
    python -m venv $venv
}

& "$venv\Scripts\Activate.ps1"
pip install -q -r $requirements

Write-Host "Environment ready. To activate it:"
Write-Host "  .venv\Scripts\Activate.ps1"
