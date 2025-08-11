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

@echo off
setlocal

REM Check if 64-bit (only AMD64)
if /I "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo System is x86_64.
) else (
    echo System is NOT x86_84. Detected architecture: %PROCESSOR_ARCHITECTURE%
    set /p choice=Do you want to proceed anyway? (y/N): 
    if /I "%choice%"=="y" (
        echo Proceeding...
    ) else (
        echo Aborting.
        exit /b 1
    )
)

endlocal

:: Path to your program (update this to match your actual file)
SET "PROGRAM_PATH=%~dp0PIRO_devices_WIN.exe"

:: Check if Git is already installed
where git >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Git is already installed.
    git --version
    GOTO CYGWIN
)

echo Git not found. Proceeding to install Git Bash...

:: Define download URL and installer path
SET "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.49.0.windows.1/Git-2.49.0-64-bit.exe"
SET "INSTALLER=%TEMP%\git-installer.exe"

:: Download Git for Windows
echo Downloading Git installer...
curl -L -o "%INSTALLER%" "%GIT_URL%"

:: Install Git silently
echo Installing Git silently...
"%INSTALLER%" /VERYSILENT /NORESTART /SUPPRESSMSGBOXES /NOCANCEL

:: Wait a bit for the system to update PATH
timeout /t 5 >nul
echo Git installed

:CYGWIN
:: Set variables
set "CYGWIN_URL=https://cygwin.com/setup-x86_64.exe"
set "INSTALLER=%TEMP%\setup-x86_64.exe"
set "INSTALL_DIR=C:\cygwin64"

:: Download Cygwin installer if not already downloaded
if not exist "%INSTALLER%" (
    echo Downloading Cygwin installer...
    curl -L -o "%INSTALLER%" "%CYGWIN_URL%"
) else (
    echo Cygwin installer already downloaded.
)

:: Run Cygwin installer silently with some common packages
echo Installing Cygwin silently...
"%INSTALLER%" -q -s http://mirrors.kernel.org/sourceware/cygwin/ -P wget,tar,bash,make,gcc-core,gcc-g++,vim,rsync -f -R "%INSTALL_DIR%"
::"%INSTALLER%" -s http://mirrors.kernel.org/sourceware/cygwin/ -P make -f -R "%INSTALL_DIR%"
SET "NewPath=C:\cygwin64\bin"
SET "Key=HKCU\Environment"
FOR /F "usebackq tokens=2*" %%A IN (`REG QUERY %Key% /v PATH 2^>nul`) DO Set CurrPath=%%B
IF NOT DEFINED CurrPath SET CurrPath=%PATH%
ECHO %CurrPath% > user_path_bak.txt
ECHO %CurrPath% | findstr /i "cygwin64\bin" >nul
IF ERRORLEVEL 1 (
    SETX PATH "%CurrPath%;%NewPath%"
    
    ECHO Cygwin added to user PATH
) ELSE (
    ECHO Cygwin already in PATH
)
SET PATH=%CurrPath%;%NewPath%
echo Cygwin installation finished.

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

echo Dependencies installed. Please run ./makedevice.sh using Git Bash.