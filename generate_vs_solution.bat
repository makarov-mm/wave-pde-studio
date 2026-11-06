@echo off
REM ---------------------------------------------------------------------------
REM Generates a Visual Studio solution with the CUDA back-ends:
REM build\WavePDEStudio.sln. CMake picks the newest Visual Studio installed.
REM
REM The Qt path and the CUDA architecture live in CMakePresets.json
REM (preset "windows-cuda"); edit them there if needed.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"
cmake --preset windows-cuda

if errorlevel 1 (
    echo.
    echo ============================================================
    echo CMake configuration FAILED.
    echo   - Check that CMAKE_PREFIX_PATH in CMakePresets.json points
    echo     to a real Qt install.
    echo   - Check that the CUDA Toolkit is installed.
    echo   - For a CPU-only build, use generate_vs_solution_cpu.bat.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Done. Open build\WavePDEStudio.sln in Visual Studio,
echo choose Release ^| x64, and build.
echo ============================================================
pause
