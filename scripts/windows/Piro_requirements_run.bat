@echo off
SETLOCAL

:: Path to your program (update this to match your actual file)
SET "PROGRAM_PATH=%~dp0PIRO_devices_WIN.exe"

:: Check if Git is already installed
where git >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo ✅ Git is already installed.
    git --version
    GOTO RUN_PROGRAM
)

echo 🔍 Git not found. Proceeding to install Git Bash...

:: Define download URL and installer path
SET "GIT_URL=https://github.com/git-for-windows/git/releases/latest/download/Git-2.45.1-64-bit.exe"
SET "INSTALLER=%TEMP%\git-installer.exe"

:: Download Git for Windows
echo ⬇️  Downloading Git installer...
powershell -Command "Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%INSTALLER%'"

:: Install Git silently
echo 🛠️  Installing Git silently...
"%INSTALLER%" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES /NOCANCEL

:: Wait a bit for the system to update PATH
timeout /t 5 >nul

:: Verify installation
where git >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo ✅ Git installed successfully.
    git --version
) ELSE (
    echo ❌ Git installation failed.
    GOTO END
)

:RUN_PROGRAM
start "" "%PROGRAM_PATH%"
echo "Please note down the Device and Platform number of the hardware where the simulation is intended to be run."
:END
echo.
pause
