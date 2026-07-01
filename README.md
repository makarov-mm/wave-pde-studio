# Wave / Heat PDE Studio — CPU vs CUDA

A real-time 2D partial-differential-equation simulator built with **Qt 6 (C++17)**.
It solves the **wave equation** and the **heat equation** on a grid with finite
differences, and runs the exact same simulation on three back-ends so you can
compare their performance live:

- **CPU (single-threaded)** — a straightforward baseline
- **CPU (multi-threaded)** — the same stencil parallelized across rows with `std::thread`
- **CUDA (GPU)** — one thread per grid cell

All three share the identical stencil math and the identical coloring
(`Stencil.h`), so their output is **pixel-for-pixel identical** — what you compare
is speed.

![screenshot](screenshot.png)

## What it demonstrates

This is a **stencil / finite-difference time-stepping** workload — a different
parallel pattern from an embarrassingly-parallel fractal or an all-pairs N-body
simulation. Each cell is updated from its four neighbors every step, and the
grid evolves over time (the state is stateful, unlike a per-pixel fractal).

- **Wave equation** `u_tt = c² ∇²u` — solved with a leapfrog scheme (keeps the
  previous field), with adjustable speed and damping. Click to drop a pulse and
  watch ripples radiate and reflect off the edges.
- **Heat equation** `u_t = α ∇²u` — solved with explicit forward Euler; paint
  heat onto the grid and watch it diffuse.

## Features

- Switch between wave and heat equations
- Selectable grid size (256² / 512² / 1024²)
- Adjustable wave speed, diffusion, damping, steps-per-frame, and palette
- **Click / drag on the canvas** to inject a disturbance
- Play / Pause / Reset
- **Benchmark all backends** — advances a fixed batch of steps on each back-end
  from the same starting field and reports ms-per-step plus the speedup relative
  to single-threaded CPU

## Requirements

- **Qt 6** (Widgets module)
- **CMake 3.20+**
- A C++17 compiler (MSVC 2022 / GCC / Clang)
- For the GPU back-end: **NVIDIA CUDA Toolkit** and an NVIDIA GPU
  (without them the project builds CPU-only and the CUDA option disappears)

## Building a Visual Studio solution (Windows, with CUDA)

CUDA on Windows requires the **MSVC** host compiler (not MinGW), so use a Qt build
for MSVC (e.g. `C:\Qt\6.x.x\msvc2022_64`).

1. Edit `generate_vs_solution.bat`, set `QT_DIR` to your MSVC Qt install, save.
2. Double-click it — it produces `build\WavePDEStudio.sln`.
   (No `-G` is passed, so CMake auto-detects the newest installed Visual Studio.)
3. Open the solution, set **WavePDEStudio** as startup project, Release | x64, build.

Command line equivalent:

```powershell
cmake -B build -S . -A x64 -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64" -DUSE_CUDA=ON -DCUDA_ARCH=89
cmake --build build --config Release
```

`CUDA_ARCH` defaults to 89 (Ada / RTX 40xx); use 86 for Ampere, 75 for Turing.

## Building without CUDA (CPU only — e.g. on macOS)

CUDA does not exist on macOS, and any machine without an NVIDIA GPU should build
CPU-only:

```bash
cmake -B build -S . -DUSE_CUDA=OFF -DCMAKE_PREFIX_PATH="/path/to/Qt/6.x.x/<kit>"
cmake --build build
```

The CUDA back-end is simply left out; the two CPU back-ends remain.

## Implementation notes

- `Types.h` and `Stencil.h` are Qt-free and compile under both a normal C++
  compiler and nvcc. The `SC_HD` macro expands to `__host__ __device__` under
  nvcc and to nothing otherwise, so one stencil and one `colorize()` run on both
  the CPU and the GPU.
- The simulation state lives on the host. The CUDA back-end uploads the field,
  iterates entirely on the device (rotating buffers between steps, no per-step
  copies), then downloads the result; the reported time is the honest wall time
  for a whole batch of steps.
- Stability is respected: the wave scheme keeps the Courant number below the 2D
  limit, and the heat scheme keeps α below 0.25.

## Possible improvements

- **Shared-memory tiling** for the stencil kernel: load a tile plus its halo into
  shared memory once per block instead of reading global memory five times per
  cell — the classic stencil optimization and a clear next step.
- Keep the field resident on the GPU while the CUDA back-end is active, uploading
  only on backend switches, to remove the per-frame transfer entirely.
- Absorbing (PML) boundaries instead of the simple zero boundary.
- 3D grids; other PDEs (reaction-diffusion, Schrödinger).

## License

MIT — do whatever you like, no warranty.
