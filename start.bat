@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if errorlevel 1 goto directory_error
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set "LAB_PYTHON="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
  if not errorlevel 1 set "LAB_PYTHON=".venv\Scripts\python.exe""
)
if defined LAB_PYTHON goto launch
py -3.12 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "LAB_PYTHON=py -3.12"
if defined LAB_PYTHON goto launch
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "LAB_PYTHON=py -3"
if defined LAB_PYTHON goto launch
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "LAB_PYTHON=python"
if defined LAB_PYTHON goto launch
echo Нужен Python 3.11 или новее.
echo Скачайте его с официального сайта: https://www.python.org/downloads/
echo В установщике включите Add Python to PATH. Затем запустите этот файл снова.
echo.
choice /c ON /m "Открыть сайт Python? O — открыть, N — закрыть"
if errorlevel 2 exit /b 1
start "" "https://www.python.org/downloads/"
exit /b 1
:launch
%LAB_PYTHON% "%~dp0run.py" %*
set "LAB_STATUS=%ERRORLEVEL%"
if not "%LAB_STATUS%"=="0" (
  echo.
  echo Запуск завершился с ошибкой. Адрес журнала указан выше.
  pause
)
exit /b %LAB_STATUS%
:directory_error
echo Не удалось открыть папку лаборатории. Распакуйте архив в доступную папку.
pause
exit /b 1
