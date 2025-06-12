@echo off
SETLOCAL EnableDelayedExpansion

:: Relaunch once in visible window
IF "%1" NEQ "fromShell" (
    echo Relaunching in visible Command Prompt window...
    cmd /k "%~f0 fromShell"
    exit /b
)
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
IF NOT '%errorlevel%'=='0' (
    echo This script requires administrator privileges.
    echo Please right-click this file and select "Run as administrator".
    goto END
)

echo Admin privileges confirmed.

:: Path to your program (update this to match your actual file)
SET "PROGRAM_PATH=%~dp0PIRO_devices_WIN.exe"

:: Check if Git is already installed
where git >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Git is already installed.
    git --version
    GOTO GPU
)

echo Git not found. Proceeding to install Git Bash...

:: Define download URL and installer path
SET "GIT_URL=https://github.com/git-for-windows/git/releases/latest/download/Git-2.45.1-64-bit.exe"
SET "INSTALLER=%TEMP%\git-installer.exe"

:: Download Git for Windows
echo Downloading Git installer...
powershell -Command "Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%INSTALLER%'"

:: Install Git silently
echo Installing Git silently...
"%INSTALLER%" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES /NOCANCEL

:: Wait a bit for the system to update PATH
timeout /t 5 >nul

:GPU
:: Check 64-bit DLL
if exist "%SystemRoot%\System32\OpenCL.dll" (
    echo OpenCL found in System32 64-bit
    goto RUN_PROGRAM
) else (
    echo OpenCL not found in System32 - Please ensure GPU drivers are installed
)

echo Detecting GPU vendor

:: Query GPU names
for /f "tokens=2 delims==" %%A in ('wmic path win32_VideoController get Name /format:list ^| findstr "Name="') do (
    set "GPUName=%%A"
)

echo GPU Detected: %GPUName%

:: Check for NVIDIA
echo %GPUName% | find /I "NVIDIA" >nul
if %errorlevel%==0 (
    echo NVIDIA GPU detected
    echo You should install the latest NVIDIA driver with OpenCL support
    start https://www.nvidia.com/Download/index.aspx
    echo "Please run this file again after installation"
    goto END
)

:: Check for AMD
echo %GPUName% | find /I "AMD" >nul
if %errorlevel%==0 (
    echo AMD GPU detected
    echo You should install the latest AMD driver
    start https://www.amd.com/en/support
    echo Please run this file again after installation
    goto END
)

:: Check for Intel
echo %GPUName% | find /I "Intel" >nul
if %errorlevel%==0 (
    echo Intel GPU detected
    echo You should install Intel Graphics drivers with OpenCL support
    start https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html
    echo Please run this file again after installation
    goto END
)

:RUN_PROGRAM
cd /d "%~dp0"
"%PROGRAM_PATH%"
echo Please note down the Device and Platform number of the hardware where the simulation is intended to be run
:END
pause
exit /b
