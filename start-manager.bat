@echo off
cd /d "%~dp0"
echo Starting llama B70 Manager server...
start "" http://127.0.0.1:31337
node server\index.js
pause
