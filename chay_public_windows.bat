@echo off
REM Chay bai tren PUBLIC TEST (1000 cau) tren may Windows co card NVIDIA (Ollama + qwen3:8b).
REM Nhap dup file nay, hoac go:  chay_public_windows.bat   trong thu muc datamate.
REM Bi dung giua chung thi chay lai: tu chay tiep cau con thieu.
REM File nop: outputs\public\nop_public.jsonl
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  echo Chua co moi truong Python. Chay truoc:  uv venv --python 3.12  va  uv pip install -r requirements.txt
  pause
  exit /b 1
)
curl -s http://localhost:11434/api/tags > nul 2>&1
if errorlevel 1 (
  echo Ollama chua chay. Mo ung dung Ollama roi chay lai file nay.
  pause
  exit /b 1
)
if not exist outputs\public mkdir outputs\public
.venv\Scripts\python.exe src\main.py --data data\public_test --out outputs\public --resume
.venv\Scripts\python.exe scripts\tao_file_nop.py data\public_test\questions.jsonl outputs\public\submission.jsonl outputs\public\nop_public.jsonl
pause
