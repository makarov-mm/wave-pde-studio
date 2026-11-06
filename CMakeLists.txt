cmake_minimum_required(VERSION 3.21)
project(WavePDEStudio LANGUAGES CXX)

# Build with CUDA acceleration by default. Turn OFF for a CPU-only build
# (the CUDA back-ends then simply disappear from the UI):  cmake -DUSE_CUDA=OFF ...
option(USE_CUDA "Build with CUDA GPU acceleration" ON)

# Target GPU architecture: 89 = Ada (RTX 40xx), 86 = Ampere (RTX 30xx),
# 75 = Turing (RTX 20xx), 120 = Blackwell (RTX 50xx). Override with -DCUDA_ARCH=86 etc.
set(CUDA_ARCH "89" CACHE STRING "CUDA compute capability")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    endif()
    enable_language(CUDA)
    # The CUDA sources need C++20 only (nvcc supports C++23 since CUDA 13.3).
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()

qt_add_executable(WavePDEStudio WIN32 MACOSX_BUNDLE
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/SimWidget.cpp
    src/SimWidget.h
    src/Simulator.cpp
    src/Simulator.h
    src/RowPool.cpp
    src/RowPool.h
    src/Types.h
    src/Stencil.h
    src/Colormap.h
)
target_link_libraries(WavePDEStudio PRIVATE Qt6::Widgets)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT WavePDEStudio)

set(msvc_cxx "$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>")
set(other_cxx "$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>")
target_compile_options(WavePDEStudio PRIVATE
    # MSVC: UTF-8 sources and strings, standard conformance, high warning level.
    "$<${msvc_cxx}:/utf-8;/permissive-;/W4>"
    # GCC/Clang: no FMA contraction, so the CPU results stay bit-identical to the GPU;
    # no FP traps (results are unaffected), so the masked stencil loops vectorize.
    "$<${other_cxx}:-Wall;-Wextra;-Wpedantic;-ffp-contract=off;-fno-trapping-math>"
)

if(USE_CUDA)
    target_sources(WavePDEStudio PRIVATE src/CudaWave.cu src/CudaWave.h)
    target_compile_definitions(WavePDEStudio PRIVATE HAVE_CUDA)
    set_target_properties(WavePDEStudio PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
    # Separate multiply and add instead of fused FMA: the same rounding as the
    # CPU code, so all back-ends produce bit-identical fields.
    target_compile_options(WavePDEStudio PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

# Copy the required Qt runtime DLLs next to the .exe on Windows (best effort).
if(WIN32)
    find_program(WINDEPLOYQT_EXE windeployqt HINTS "${Qt6_DIR}/../../../bin")
    if(WINDEPLOYQT_EXE)
        add_custom_command(TARGET WavePDEStudio POST_BUILD
            COMMAND "${WINDEPLOYQT_EXE}" --no-translations "$<TARGET_FILE:WavePDEStudio>"
            COMMENT "Deploying Qt runtime with windeployqt")
    endif()
endif()
