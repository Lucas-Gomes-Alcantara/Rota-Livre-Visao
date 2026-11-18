@echo off
cd /d "%~dp0"
set LOG=results\logs\enviar_docs.log
git add datasets\README.md enviar_docs.bat > %LOG% 2>&1
git commit -m "Documenta numeros reais das bases, forks e inspecao visual" >> %LOG% 2>&1
git push >> %LOG% 2>&1
echo [git] push exit=%errorlevel% >> %LOG%
git status --short >> %LOG% 2>&1
