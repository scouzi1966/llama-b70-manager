@echo off
cd /d "%~dp0"
echo Starting llama B70 Manager...
start "" http://127.0.0.1:31337
npm run dev
pause
