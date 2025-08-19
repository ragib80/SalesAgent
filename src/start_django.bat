@echo off
setlocal

REM ==== Force UTF-8 so logs don't crash on Unicode ====
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

REM ==== Paths & settings ====
set "PROJECT_DIR=C:\Users\ragib\DeployInternally\SalesAgent\src"
set "PYTHON_EXE=C:\Users\ragib\DeployInternally\SalesAgent\venv\Scripts\python.exe"
set "HOST=0.0.0.0"
set "PORT=9000"
REM set "DJANGO_SETTINGS_MODULE=mysite.settings"   REM <- if you use a custom settings module

REM ==== Logs ====
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

pushd "%PROJECT_DIR%"

REM Identify who is running (useful for debugging Scheduled Task identity)
echo [START %DATE% %TIME%] Launching Django on %HOST%:%PORT% >> "%PROJECT_DIR%\logs\django.log"
echo USERNAME=%USERNAME% >> "%PROJECT_DIR%\logs\django.log"
whoami >> "%PROJECT_DIR%\logs\django.log"

REM ==== Optional boot tasks ====
REM "%PYTHON_EXE%" -X utf8 manage.py migrate --noinput >> "%PROJECT_DIR%\logs\django.log" 2>&1
REM "%PYTHON_EXE%" -X utf8 manage.py collectstatic --noinput >> "%PROJECT_DIR%\logs\django.log" 2>&1

REM ==== Run Django dev server ====
"%PYTHON_EXE%" -X utf8 manage.py runserver %HOST%:%PORT% --insecure >> "%PROJECT_DIR%\logs\django.log" 2>&1

popd
endlocal
