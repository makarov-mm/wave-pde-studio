# Wave / Heat PDE Studio — CPU vs CUDA

[![GitHub Sponsors](https://img.shields.io/github/sponsors/makarov-mm?style=flat&logo=github)](https://github.com/sponsors/makarov-mm)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B-00599C)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/makarov-mm/)
[![Threads](https://img.shields.io/badge/Threads-000000?logo=threads&logoColor=white)](https://www.threads.net/@m.m.makarov)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/m.m.makarov/)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

A real-time 2D partial-differential-equation simulator built with **Qt 6 (C++23)**.
It solves the **wave equation**, the **heat equation**, and the **Gray–Scott
reaction–diffusion system** on a grid with finite differences, and runs the exact
same simulation on four back-ends so you can compare their performance live:

- **CPU (single-threaded)** — a straightforward baseline
- **CPU (multi-threaded)** — the same stencil parallelized across rows with a persistent thread pool
- **CUDA (GPU)** — one thread per grid cell, global-memory reads
- **CUDA (GPU, tiled)** — one thread per cell, but each block first stages an
  18×18 tile (16×16 plus a one-cell halo) into shared memory, then computes the
  Laplacian from the tile

All back-ends call the same cell-update functions (`Stencil.h`) and are built
without fused multiply-add, so they produce **bit-identical** fields — what you
compare is speed.

![screenshot](screenshot.png)

## What it demonstrates

This is a **stencil / finite-difference time-stepping** workload — a different
parallel pattern from an embarrassingly-parallel fractal or an all-pairs N-body
simulation. Each cell is updated from its four neighbors every step, the state
is carried between steps, and the memory-access pattern (each value read by five
threads) is exactly the case shared-memory tiling was designed for.

### Equations

- **Wave equation** `u_tt = c² ∇²u` — leapfrog scheme (keeps the previous
  field), adjustable speed and damping. Click to drop a pulse and watch ripples
  radiate, reflect, and diffract.
- **Heat equation** `u_t = α ∇²u` — explicit forward Euler; paint heat onto the
  grid and watch it diffuse.
- **Gray–Scott reaction–diffusion** — two coupled fields *u*, *v* with
  `u_t = Dᵤ∇²u − uv² + F(1−u)` and `v_t = Dᵥ∇²v + uv² − (F+k)v`.
  Depending on the feed/kill rates it self-organizes into corals, spots,
  worms, dividing cells, or chaos. Preset patterns included
  (**Coral, Mitosis, Worms, Spots, Chaos**) plus free Feed/Kill sliders.

### Boundary conditions

Selectable per run, implemented once in `Stencil.h` and used by both CPU and
CUDA paths:

- **Dirichlet** (fixed edges — waves reflect with inversion)
- **Neumann** (reflective — zero normal derivative, waves reflect without inversion)
- **Periodic** (torus — waves leaving on one side re-enter on the other)

### Obstacles and scenes

Any cell can be marked as a **wall**: walls are pinned to the rest value each
step, so waves reflect off them and Gray–Scott patterns grow around them.
Paint walls with the **right mouse button**, erase with **Shift + right button**.

Built-in scenes rebuild the wall mask on demand:

- **Double slit** — a wall with two gaps; drop a pulse behind it and watch the
  classic interference pattern form
- **Ring cavity** — a circular resonator with a small opening
- **Pillar lattice** — an 8×8 array of circular pillars (a crude photonic crystal)

### Rendering

Five palettes: **Thermal (diverging)**, **Ocean (diverging)**, **Grayscale**,
**Viridis**, and **Magma** — the last two are piecewise-linear LUT
approximations of the matplotlib colormaps. The grid is shown with its aspect
ratio preserved, and the current frame can be saved with **Export PNG**.

## The four back-ends

| Backend | Idea |
|---|---|
| CPU (single) | Plain loop over rows/columns; interior cells skip the boundary logic, so the compiler can vectorize the hot loop |
| CPU (multi) | Rows split into chunks, distributed over a **persistent thread pool** (workers park on a condition variable between substeps — no thread creation per step) |
| CUDA (naive) | One thread per cell, neighbors read from global memory |
| CUDA (tiled) | Each 16×16 block stages an 18×18 tile into `__shared__` memory, halo included, then computes from the tile — each global value is loaded once per block instead of up to five times |

The **Benchmark all backends** button runs one warm-up step (thread start-up,
CUDA context and allocations stay out of the numbers), then 100 identical steps
on each back-end, and reports ms/step and the speed-up factor. The live timing
panel additionally shows an exponential moving average and throughput in
**Mcells/s**. GPU timings include the host↔device transfers.

If the GPU can't run the kernels (no driver, or a build for another compute
capability), the CUDA back-ends are hidden and the reason is shown in the panel.

## Controls

| Action | Input |
|---|---|
| Inject pulse / heat / chemical | Left mouse drag |
| Paint wall | Right mouse drag |
| Erase wall | Shift + right mouse drag |
| Pause / resume | `Space` |
| Reset | `R` |
| Brush size / power | Sliders in the panel |

## Building

Requirements: **CMake ≥ 3.21**, **Qt 6** (Widgets), a **C++23** compiler
(Visual Studio 2022/2026, GCC 13+, Clang 18+), and optionally the **CUDA toolkit**
(12.x or 13.x; the CUDA sources need C++20 only).

```bash
# with CUDA (set your GPU architecture; 89 = Ada / RTX 40xx)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCUDA_ARCH=89
cmake --build build --config Release

# CPU-only (no CUDA toolkit needed; the two CUDA back-ends are hidden)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
cmake --build build --config Release
```

On Windows, `CMakePresets.json` holds the Qt path (`CMAKE_PREFIX_PATH`) and the
CUDA architecture; edit them there or override them in a `CMakeUserPresets.json`.
Then either run `generate_vs_solution.bat` / `generate_vs_solution_cpu.bat` to
get a Visual Studio solution, or use the presets directly:

```bat
cmake --preset windows-cuda
cmake --build --preset cuda-release
```

## Project layout

```
src/
  Types.h          — SimParams and the enums (equation, boundary, scene, palette)
  Stencil.h        — shared cell math; compiled as C++ AND as CUDA device code
  Colormap.h       — palettes, field value -> pixel
  RowPool.h/.cpp   — persistent thread pool for the multi-threaded back-end
  Simulator.h/.cpp — grid state, wall mask, scenes, CPU back-ends
  CudaWave.h/.cu   — CUDA back-ends (naive + shared-memory tiled), device buffers
  MainWindow.*     — UI, benchmark, presets, export
  SimWidget.*      — canvas, mouse painting
  main.cpp
```

## Notes on correctness

Every back-end evaluates the same `Stencil.h` functions with the same operand
order. Fused multiply-add is disabled where a compiler would otherwise use it
(`--fmad=false` for nvcc, `-ffp-contract=off` for GCC/Clang; MSVC doesn't contract
by default), because an FMA rounds differently from a separate multiply and add.
As a result, the single-threaded, multi-threaded and both CUDA back-ends produce
bit-identical fields for every equation, boundary condition and scene.

Updated values smaller than 1e-12 are stored as zero. They are far below anything
visible, and dropping them keeps the arithmetic out of the subnormal-float range,
which otherwise slows the CPU back-ends down several times (Gray–Scott's fading
*v* field, decaying waves).

## License

MIT License

Copyright (c) 2026 Mykhailo Makarov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

If you found this project interesting or useful, you can support my work:

[![GitHub Sponsors](https://img.shields.io/github/sponsors/makarov-mm?style=flat&logo=github)](https://github.com/sponsors/makarov-mm)
