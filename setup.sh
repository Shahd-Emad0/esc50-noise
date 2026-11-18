#!/usr/bin/env bash
# Linux/macOS: bash setup.sh        (bash setup.sh --quick for a 10-minute smoke test)
set -e
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then $PY -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
else $PY -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121; fi
$PY -m pip install -r requirements.txt
$PY run_all.py "$@"
