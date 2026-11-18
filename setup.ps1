# Windows: powershell -ExecutionPolicy Bypass -File setup.ps1        (add -Quick for a 10-minute smoke test)
param([switch]$Quick)
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& $py -m pip install -r requirements.txt
if ($Quick) { & $py run_all.py --quick } else { & $py run_all.py }
