@echo off
setlocal EnableDelayedExpansion

REM ==== Django Service Startup Script ====
REM This script is designed to run as a Windows Service via NSSM

REM ==== Force UTF-8 so logs don't crash on Unicode ====
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ==== Paths & settings ====
set "PROJECT_DIR=C:\Users\ragib\DeployInternally\SalesAgent\src"
set "VENV_DIR=C:\Users\ragib\DeployInternally\SalesAgent\venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "ACTIVATE_SCRIPT=%VENV_DIR%\Scripts\activate.bat"
set "HOST=0.0.0.0"
set "PORT=9000"

REM ==== SSL certificate paths ====
set "SSL_DIR=C:\Users\ragib\DeployInternally\ssl"
set "SSL_CERT=%SSL_DIR%\BPBL-2025-2026.crt"
set "SSL_KEY=%SSL_DIR%\BPBL-2025-2026.key"

REM ==== Create logs directory if it doesn't exist ====
if not exist "%PROJECT_DIR%\logs" (
    mkdir "%PROJECT_DIR%\logs" 2>nul
)

REM ==== Change to project directory ====
cd /d "%PROJECT_DIR%" 2>nul
if errorlevel 1 (
    echo ERROR: Cannot change to project directory: %PROJECT_DIR%
    echo ERROR: Cannot change to project directory: %PROJECT_DIR% >> "%PROJECT_DIR%\logs\service.log" 2>&1
    exit /b 1
)

REM ==== Log startup info ====
set "TIMESTAMP=%DATE% %TIME%"
echo ================================================== >> "%PROJECT_DIR%\logs\service.log"
echo [SERVICE START %TIMESTAMP%] Django Service Starting >> "%PROJECT_DIR%\logs\service.log"
echo USERNAME=%USERNAME% >> "%PROJECT_DIR%\logs\service.log"
echo COMPUTERNAME=%COMPUTERNAME% >> "%PROJECT_DIR%\logs\service.log"
echo PROJECT_DIR=%PROJECT_DIR% >> "%PROJECT_DIR%\logs\service.log"
echo VENV_DIR=%VENV_DIR% >> "%PROJECT_DIR%\logs\service.log"
echo PYTHON_EXE=%PYTHON_EXE% >> "%PROJECT_DIR%\logs\service.log"
echo SSL_CERT=%SSL_CERT% >> "%PROJECT_DIR%\logs\service.log"
echo SSL_KEY=%SSL_KEY% >> "%PROJECT_DIR%\logs\service.log"
echo ================================================== >> "%PROJECT_DIR%\logs\service.log"

REM ==== Verify critical files exist ====
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found: %PYTHON_EXE% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

if not exist "%PROJECT_DIR%\manage.py" (
    echo ERROR: Django manage.py not found in: %PROJECT_DIR% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

if not exist "%ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment activation script not found: %ACTIVATE_SCRIPT% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

if not exist "%SSL_CERT%" (
    echo ERROR: SSL certificate not found: %SSL_CERT% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

if not exist "%SSL_KEY%" (
    echo ERROR: SSL key not found: %SSL_KEY% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

REM ==== Activate virtual environment ====
echo Activating virtual environment... >> "%PROJECT_DIR%\logs\service.log"
call "%ACTIVATE_SCRIPT%" >> "%PROJECT_DIR%\logs\service.log" 2>&1

REM ==== Verify django-extensions is available ====
"%PYTHON_EXE%" -m pip show django-extensions >nul 2>&1
if errorlevel 1 (
    echo ERROR: django-extensions not installed in venv. >> "%PROJECT_DIR%\logs\service.log"
    echo Run: "%PYTHON_EXE%" -m pip install django-extensions >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

REM ==== Verify virtual environment activation ====
echo Testing Python and virtual environment... >> "%PROJECT_DIR%\logs\service.log"
"%PYTHON_EXE%" -c "import sys; print('Python executable:', sys.executable)" >> "%PROJECT_DIR%\logs\service.log" 2>&1
"%PYTHON_EXE%" -c "import django; print('Django version:', django.get_version())" >> "%PROJECT_DIR%\logs\service.log" 2>&1

REM ==== Preload pyOpenSSL to ensure it's available ====
"%PYTHON_EXE%" -c "import OpenSSL; print('pyOpenSSL OK:', OpenSSL.__version__)" >> "%PROJECT_DIR%\logs\service.log" 2>&1
if errorlevel 1 (
    echo ERROR: pyOpenSSL not available in venv. >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

REM ==== Start Django development server with SSL ====
echo Starting Django server on %HOST%:%PORT% with SSL >> "%PROJECT_DIR%\logs\service.log"
echo [INFO %TIMESTAMP%] Django server starting on %HOST%:%PORT% with SSL >> "%PROJECT_DIR%\logs\service.log"

"%PYTHON_EXE%" manage.py runserver_plus %HOST%:%PORT% --cert-file "%SSL_CERT%" --key-file "%SSL_KEY%" --insecure

REM ==== Handle exit code ====
set "EXIT_CODE=%ERRORLEVEL%"
set "TIMESTAMP=%DATE% %TIME%"
echo [STOP %TIMESTAMP%] Django server stopped with exit code: %EXIT_CODE% >> "%PROJECT_DIR%\logs\service.log"

exit /b %EXIT_CODE%
