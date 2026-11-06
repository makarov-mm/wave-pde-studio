@echo off
REM ---------------------------------------------------------------------------
REM Generates a CPU-only Visual Studio solution (no CUDA toolkit needed):
REM build-cpu\WavePDEStudio.sln. The two CPU back-ends remain in the UI.
REM
REM The Qt path lives in CMakePresets.json (preset "windows-cpu").
REM ---------------------------------------------------------------------------

cd /d "%~dp0"
cmake --preset windows-cpu

if errorlevel 1 (
    echo.
    echo CMake configuration FAILED. Check that CMAKE_PREFIX_PATH in
    echo CMakePresets.json points to a real Qt install.
    pause
    exit /b 1
)

echo.
echo Done. Open build-cpu\WavePDEStudio.sln, choose Release ^| x64, and build.
pause
