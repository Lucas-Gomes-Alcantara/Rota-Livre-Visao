@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set LOG=results\logs\rodar_parte3.log
echo ==== inicio %date% %time% ==== > %LOG%
set PY=python
if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
for %%S in ("src\download_data.py roboflow" "src\convert.py" "src\split.py" "src\check_dataset.py" "src\visualize_labels.py --per-class 12") do (
  echo --- %%~S
  echo. >> %LOG%
  echo ==== %%~S ==== >> %LOG%
  %PY% -u %%~S >> %LOG% 2>&1
  echo [exit] %%~S = !errorlevel! >> %LOG%
)
echo ==== git ==== >> %LOG%
git add src rodar_parte3.bat tools >> %LOG% 2>&1
git commit -m "Download de forks do Roboflow sem versao publicada; Mapillary mais eficiente" >> %LOG% 2>&1
git push >> %LOG% 2>&1
echo [git] push exit=!errorlevel! >> %LOG%
echo --- mapillary
echo ==== mapillary ==== >> %LOG%
%PY% -u src\mapillary_fetch.py --city sao_paulo rio_de_janeiro belo_horizonte --n 300 --require-sidewalk >> %LOG% 2>&1
echo [exit] mapillary = !errorlevel! >> %LOG%
echo ==== fim %date% %time% ==== >> %LOG%
echo Concluido. Log em %LOG%
pause
