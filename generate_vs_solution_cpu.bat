@echo off
REM ---------------------------------------------------------------------------
REM Generates a CPU-only Visual Studio solution (no CUDA required).
REM The CUDA back-end is left out; the two CPU back-ends remain in the UI.
REM ---------------------------------------------------------------------------

REM Change this to your Qt path:
set QT_DIR=C:\Qt\6.7.2\msvc2022_64

cmake -B build-cpu -S . -A x64 -DCMAKE_PREFIX_PATH="%QT_DIR%" -DUSE_CUDA=OFF

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CMake configuration FAILED. Check that QT_DIR points to a real Qt install.
    pause
    exit /b 1
)

echo.
echo Done. Open build-cpu\WavePDEStudio.sln in Visual Studio,
echo choose Release ^| x64, and build.
pause
