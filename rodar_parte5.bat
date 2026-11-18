@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set LOG=results\logs\rodar_parte5.log
echo ==== inicio %date% %time% ==== > %LOG%
set PY=python
if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
echo ==== GPU ==== >> %LOG%
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> %LOG% 2>&1
echo [gpu] exit=!errorlevel! >> %LOG%
%PY% -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" >> %LOG% 2>&1
echo --- mapillary
echo ==== mapillary ==== >> %LOG%
%PY% -u src\mapillary_fetch.py --city sao_paulo rio_de_janeiro belo_horizonte --n 300 --per-sequence 2 --out data\raw\mapillary_br_v2 >> %LOG% 2>&1
echo [exit] mapillary = !errorlevel! >> %LOG%
%PY% tools\zip_para_anotar.py data\raw\mapillary_br_v2 mapillary_br_para_anotar.zip >> %LOG% 2>&1
echo ==== git ==== >> %LOG%
git add configs src tools rodar_parte5.bat >> %LOG% 2>&1
git commit -m "Mapillary sem filtro de calcada (triagem na anotacao); empacotamento para anotar" >> %LOG% 2>&1
git push >> %LOG% 2>&1
echo [git] push exit=!errorlevel! >> %LOG%
echo ==== fim %date% %time% ==== >> %LOG%
echo Concluido. Log em %LOG%
pause
