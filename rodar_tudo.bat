@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist results\logs mkdir results\logs
set LOG=results\logs\rodar_tudo.log
echo ==== inicio %date% %time% ==== > %LOG%

set PY=python
if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
echo [python] %PY% >> %LOG%
%PY% --version >> %LOG% 2>&1

echo.
echo ===== 1/3 Enviando codigo para o GitHub =====
where git >> %LOG% 2>&1
if errorlevel 1 (
  echo [git] GIT NAO ENCONTRADO >> %LOG%
  echo Git nao encontrado - pulando envio ao GitHub.
  goto DADOS
)
if not exist .git git init >> %LOG% 2>&1
git config user.name >nul 2>&1 || git config user.name "Lucas Gomes Alcantara"
git config user.email >nul 2>&1 || git config user.email "Lucas-Gomes-Alcantara@users.noreply.github.com"
git add . >> %LOG% 2>&1
echo [git] arquivos no commit: >> %LOG%
git diff --cached --name-only >> %LOG% 2>&1
git diff --cached --name-only | findstr /I /C:".env" /C:".zip" >nul && (
  echo [git] ABORTADO: .env ou .zip entraria no commit >> %LOG%
  echo ERRO: .env ou zip entraria no commit. Abortando envio.
  goto DADOS
)
git commit -m "Pipeline Rota Livre: dados, treino, avaliacao e demo" >> %LOG% 2>&1
git branch -M main >> %LOG% 2>&1
git remote get-url origin >nul 2>&1 || git remote add origin https://github.com/Lucas-Gomes-Alcantara/Rota-Livre-Visao.git
git push -u origin main >> %LOG% 2>&1
echo [git] push exit=%errorlevel% >> %LOG%

:DADOS
echo.
echo ===== 2/3 Instalando dependencias da etapa de dados =====
%PY% -m pip install -q -r requirements-dados.txt >> %LOG% 2>&1
echo [pip] exit=%errorlevel% >> %LOG%

echo.
echo ===== 3/3 Etapa de dados =====
for %%S in ("src\download_data.py roboflow" "src\convert.py" "src\mapillary_fetch.py --city sao_paulo rio_de_janeiro belo_horizonte --n 300 --require-sidewalk" "src\split.py" "src\check_dataset.py" "src\visualize_labels.py --per-class 12") do (
  echo --- %%~S
  echo. >> %LOG%
  echo ==== %%~S ==== >> %LOG%
  %PY% %%~S >> %LOG% 2>&1
  echo [exit] %%~S = !errorlevel! >> %LOG%
)
echo ==== fim %date% %time% ==== >> %LOG%
echo.
echo Concluido. Log em %LOG%
pause
