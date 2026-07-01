@echo off
REM ---------------------------------------------------------------------------
REM Generates a Visual Studio solution (build\WavePDEStudio.sln) using CMake.
REM CMake configures Qt's MOC step, the CUDA build and all paths automatically.
REM
REM 1) Set QT_DIR below to your Qt installation.
REM 2) Double-click this file.
REM 3) Open build\WavePDEStudio.sln, pick Release ^| x64, build.
REM ---------------------------------------------------------------------------

REM Change this to your Qt path (the folder that contains bin\, lib\, include\):
set QT_DIR=C:\Qt\6.7.2\msvc2022_64

REM CUDA architecture: 89 = Ada (RTX 40xx), 86 = Ampere (RTX 30xx), 75 = Turing.
set CUDA_ARCH=89

REM -A x64 selects the 64-bit platform. No -G is passed, so CMake auto-detects
REM the newest Visual Studio installed (works for VS 2022 and VS 2026).
cmake -B build -S . -A x64 -DCMAKE_PREFIX_PATH="%QT_DIR%" -DUSE_CUDA=ON -DCUDA_ARCH=%CUDA_ARCH%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo CMake configuration FAILED.
    echo   - Check that QT_DIR above points to a real Qt install.
    echo   - Check that the CUDA Toolkit is installed and on PATH.
    echo   - For a CPU-only build, use generate_vs_solution_cpu.bat.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Done. Open build\WavePDEStudio.sln in Visual Studio,
echo set WavePDEStudio as the startup project, choose Release ^| x64,
echo and build.
echo ============================================================
pause
