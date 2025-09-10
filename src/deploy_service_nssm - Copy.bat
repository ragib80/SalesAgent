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
echo ================================================== >> "%PROJECT_DIR%\logs\service.log"

REM ==== Verify critical files exist ====
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found: %PYTHON_EXE% >> "%PROJECT_DIR%\logs\service.log"
    echo ERROR: Python executable not found: %PYTHON_EXE%
    echo Make sure virtual environment is properly installed >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

if not exist "%PROJECT_DIR%\manage.py" (
    echo ERROR: Django manage.py not found in: %PROJECT_DIR% >> "%PROJECT_DIR%\logs\service.log"
    echo ERROR: Django manage.py not found in: %PROJECT_DIR%
    exit /b 1
)

if not exist "%ACTIVATE_SCRIPT%" (
    echo ERROR: Virtual environment activation script not found: %ACTIVATE_SCRIPT% >> "%PROJECT_DIR%\logs\service.log"
    echo ERROR: Virtual environment activation script not found: %ACTIVATE_SCRIPT%
    exit /b 1
)

REM ==== Activate virtual environment ====
echo Activating virtual environment... >> "%PROJECT_DIR%\logs\service.log"
call "%ACTIVATE_SCRIPT%" >> "%PROJECT_DIR%\logs\service.log" 2>&1

REM ==== Verify virtual environment activation ====
echo Testing Python and virtual environment... >> "%PROJECT_DIR%\logs\service.log"
"%PYTHON_EXE%" -c "import sys; print('Python executable:', sys.executable)" >> "%PROJECT_DIR%\logs\service.log" 2>&1
"%PYTHON_EXE%" -c "import django; print('Django version:', django.get_version())" >> "%PROJECT_DIR%\logs\service.log" 2>&1

if errorlevel 1 (
    echo ERROR: Django not properly installed in virtual environment >> "%PROJECT_DIR%\logs\service.log"
    echo ERROR: Django not properly installed in virtual environment
    echo Check that Django is installed in: %VENV_DIR% >> "%PROJECT_DIR%\logs\service.log"
    exit /b 1
)

REM ==== Run Django setup tasks (optional - uncomment if needed) ====
echo Running Django setup tasks... >> "%PROJECT_DIR%\logs\service.log"
REM "%PYTHON_EXE%" manage.py migrate --noinput >> "%PROJECT_DIR%\logs\service.log" 2>&1
REM "%PYTHON_EXE%" manage.py collectstatic --noinput >> "%PROJECT_DIR%\logs\service.log" 2>&1

REM ==== Start Django development server ====
echo Starting Django server on %HOST%:%PORT% >> "%PROJECT_DIR%\logs\service.log"
echo [INFO %TIMESTAMP%] Django server starting on %HOST%:%PORT% >> "%PROJECT_DIR%\logs\service.log"

REM Start Django with explicit Python path from virtual environment
"%PYTHON_EXE%" manage.py runserver %HOST%:%PORT% --insecure

REM If we get here, the server stopped
set "EXIT_CODE=%ERRORLEVEL%"
set "TIMESTAMP=%DATE% %TIME%"
echo [STOP %TIMESTAMP%] Django server stopped with exit code: %EXIT_CODE% >> "%PROJECT_DIR%\logs\service.log"

exit /b %EXIT_CODE%