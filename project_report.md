# Mandelbrot Set Renderer - Complete Project Report

## Executive Summary

This project is a comprehensive Mandelbrot set visualization tool that compares CPU and GPU rendering performance, provides interactive exploration capabilities, and generates high-quality images and animations with advanced rendering effects. The system is built in Python and leverages CUDA GPU acceleration through CuPy for significant performance improvements.

**Key Achievements:**
- 2-3x GPU speedup over CPU rendering
- Interactive dual-view explorer with synchronized navigation
- Advanced rendering with stripe shading, step shading, and Blinn-Phong lighting
- Automated performance analysis and visualization
- Preset system for interesting fractal regions
- High-resolution output (up to 2560×1440)

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Core Modules](#core-modules)
3. [Function Reference](#function-reference)
4. [Data Structures](#data-structures)
5. [Rendering Pipeline](#rendering-pipeline)
6. [Performance Metrics](#performance-metrics)
7. [User Workflows](#user-workflows)
8. [Technical Implementation](#technical-implementation)

---

## Project Architecture

### File Structure

```
files (1)/
├── compare_cpu_gpu.py          # Main application (1134 lines)
├── mandelbrot_cpu_only.py      # CPU renderer (7940 bytes)
├── mandelbrot_gpu.py           # GPU renderer (18367 bytes)
├── make_animation_advanced.py  # Animation generator (12433 bytes)
├── advanced_mandelbrot.py      # Advanced effects (18414 bytes)
├── coordinate_presets.json     # Saved presets (3148 bytes)
├── requirements.txt            # Dependencies
├── README.md                   # User documentation
├── project_report.md           # Technical documentation
├── images/                     # Organized by resolution
│   ├── 800x600/               # Standard resolution comparisons
│   ├── 1920x1080/             # Full HD comparisons
│   ├── 2560x1440/             # QHD/2K comparisons
│   ├── crown.png
│   ├── julia.png
│   ├── lightning.png
│   ├── octogone.png
│   ├── pow.png
│   ├── spiral.png
│   ├── tiles.png
│   ├── velvet.png
│   └── web.png
└── gifs/                       # Animations output
    └── zoom_advanced.gif
```

### Module Dependencies

```
compare_cpu_gpu.py
├── mandelbrot_cpu_only.py
│   └── numba (JIT compilation)
├── mandelbrot_gpu.py
│   └── cupy (CUDA acceleration)
├── numpy (numerical operations)
├── matplotlib (visualization)
└── PIL (image saving)

make_animation_advanced.py
├── mandelbrot_gpu.py
├── imageio (GIF creation)
└── numpy

advanced_mandelbrot.py
├── numpy
└── numba
```

---

## Core Modules

### 1. compare_cpu_gpu.py (Main Application)

**Purpose:** Unified comparison tool with menu-driven interface

**Classes:**

#### `ComparisonMetrics` (Dataclass)
Stores performance metrics across multiple runs.

**Attributes:**
- `iteration_counts: List[int]` - Iteration counts tested
- `cpu_times: List[float]` - CPU computation times
- `gpu_times: List[float]` - GPU computation times
- `cpu_throughputs: List[float]` - CPU pixels/second
- `gpu_throughputs: List[float]` - GPU pixels/second
- `speedups: List[float]` - GPU speedup factors

#### `ComparisonTool` (Main Class)
Orchestrates all comparison and rendering operations.

**Attributes:**
- `width: int` - Image width (default: 800)
- `height: int` - Image height (default: 600)
- `images_dir: str` - Output directory for images
- `gifs_dir: str` - Output directory for animations
- `metrics: ComparisonMetrics` - Performance data
- `interactive_metrics: List[dict]` - Interactive session data

**Methods:**

##### `__init__(width=800, height=600)`
Initialize the comparison tool with specified dimensions.

##### `apply_color_scheme(M, max_iter, rgb_thetas=(0.85, 0.0, 0.15))`
Apply sinusoidal color mapping to iteration matrix.
- **Input:** Iteration matrix, max iterations, RGB phase offsets
- **Output:** RGB image array (uint8)
- **Algorithm:** Sinusoidal color table with configurable phases

##### `side_by_side_comparison()`
Compare CPU vs GPU at multiple iteration counts (256, 512, 1024, 2048).
- **Resolution Selection:** User chooses 800×600, 1920×1080, or 2560×1440
- **Output:** 4 comparison images saved in resolution-specific subfolders
- **Layout:** Dynamically adjusted for each resolution (2×2 grid with vertical summary)
- **Metrics:** Time, throughput, speedup for each iteration count
- **Subfolder Structure:** `images/[resolution]/comparison_[iterations]_iterations.png`
- **Layout Optimization:**
  - 800×600: Figure 10×6", title 12pt, text 8pt
  - 1920×1080: Figure 12.8×7.2", title 14pt, text 10pt
  - 2560×1440: Figure 16×9", title 16pt, text 11pt

##### `dual_interactive_explorer()`
Launch synchronized CPU/GPU explorers with zoom/pan.
- **Resolution:** Fixed 800×600 for optimal interactive performance
- **Controls:** Left-click zoom in, right-click zoom out
- **Synchronization:** Both views maintain identical coordinates
- **Metrics Collection:** Records time for each interaction
- **Auto-Analysis:** Opens comparative analysis on close

##### `comparative_analysis()`
Generate performance analysis graphs from interactive session.
- **4-Panel Layout:**
  1. Render time per interaction (line chart)
  2. GPU speedup per interaction (bar chart)
  3. Cumulative time comparison (line chart)
  4. Summary statistics (text box)
- **Output:** `interactive_performance_analysis.png`

##### `create_advanced_animation()`
Generate zoom animation with advanced rendering effects.
- **Parameters:** RGB thetas, stripe/step frequencies, lighting
- **Frames:** 150 frames (configurable)
- **Zoom:** Logarithmic scale from 3.0 to 0.000001
- **Target:** (-1.749705768080503, -6.13369029080495e-05)
- **Output:** GIF in `gifs/` directory

##### `render_custom_coordinates()`
Render specific Mandelbrot regions with preset support.
- **Modes:**
  1. Use preset (9 available)
  2. Enter custom coordinates
  3. Save as preset
- **Advanced Options:** Stripe/step shading, lighting
- **Output:** 1920×1080 PNG in `images/` directory

##### `run_all()`
Execute options 1-3 sequentially for comprehensive analysis.

**Helper Functions:**

##### `sin_colortable_advanced(rgb_thetas, ncol=2**12)`
Create sinusoidal color table for advanced rendering.
- **Input:** RGB phase offsets, number of colors
- **Output:** Color table array (ncol × 3)

##### `blinn_phong(normal, light)` [@jit]
Blinn-Phong shading algorithm for 3D lighting effects.
- **Input:** Surface normal (complex), light parameters (7-element array)
- **Output:** Brightness value (float)
- **Components:** Diffuse + specular lighting

##### `smooth_iter(c, maxiter, stripe_s, stripe_sig)` [@jit]
Compute smooth iteration count with stripe data.
- **Input:** Complex point, max iterations, stripe parameters
- **Output:** (iteration, stripe_avg, DEM, normal)
- **Algorithm:** Escape time with continuous potential

##### `color_pixel_advanced(matxy, niter, stripe_a, step_s, dem, normal, colortable, ncycle, light)` [@jit]
Color pixel with advanced effects.
- **Input:** Pixel array, iteration data, rendering parameters
- **Output:** Modifies matxy in-place with RGB values
- **Effects:** Overlay blending, stripe/step shading, lighting

##### `compute_set_advanced(creal, cim, maxiter, colortable, ncycle, stripe_s, stripe_sig, step_s, diag, light)` [@jit]
Compute Mandelbrot set with advanced rendering.
- **Input:** Coordinate arrays, rendering parameters
- **Output:** RGB image array (height × width × 3)
- **Optimization:** Numba JIT compilation

---

### 2. mandelbrot_cpu_only.py (CPU Renderer)

**Purpose:** CPU-only Mandelbrot rendering with Numba acceleration

**Classes:**

#### `RenderMetrics` (Dataclass)
Store rendering performance metrics.

**Attributes:**
- `time_taken: float` - Computation time (seconds)
- `throughput: float` - Pixels per second
- `memory_used: float` - Memory usage (bytes)

#### `MandelbrotCPU`
Basic CPU renderer with progress tracking.

**Attributes:**
- `width, height: int` - Image dimensions
- `max_iter: int` - Maximum iterations
- `metrics: RenderMetrics` - Performance data

**Methods:**

##### `__init__(width, height, max_iter)`
Initialize CPU renderer.

##### `compute(xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25)`
Compute Mandelbrot set on CPU.
- **Algorithm:** Escape time algorithm with Numba JIT
- **Progress:** Prints every 32 iterations
- **Output:** Iteration matrix (height × width)
- **Metrics:** Automatically calculated and stored

##### `mandelbrot_kernel(c, max_iter)` [@jit]
Core Mandelbrot computation kernel.
- **Input:** Complex number, max iterations
- **Output:** Iteration count at escape
- **Escape Radius:** 2.0

#### `InteractiveMandelbrotCPU`
Interactive CPU explorer with zoom/pan capabilities.

**Methods:**

##### `run()`
Launch interactive matplotlib window.
- **Controls:** Left-click zoom in, right-click zoom out
- **Display:** Real-time rendering with colormap
- **Title:** Shows computation time

---

### 3. mandelbrot_gpu.py (GPU Renderer)

**Purpose:** GPU-accelerated rendering with CPU fallback

**Classes:**

#### `GPUMetrics` (Dataclass)
GPU-specific performance metrics.

**Attributes:**
- `time_taken: float` - GPU computation time
- `throughput: float` - Pixels per second
- `memory_used: float` - GPU memory usage
- `device_name: str` - GPU device identifier

#### `MandelbrotRenderer`
Unified renderer with GPU/CPU support.

**Attributes:**
- `width, height: int` - Image dimensions
- `max_iter: int` - Maximum iterations
- `gpu_metrics: GPUMetrics` - GPU performance data
- `cpu_metrics: RenderMetrics` - CPU performance data (fallback)

**Methods:**

##### `__init__(width, height, max_iter)`
Initialize renderer with GPU detection.

##### `mandelbrot_gpu(xmin, xmax, ymin, ymax)`
Compute on GPU using CuPy.
- **Kernel:** Custom CUDA kernel for Mandelbrot computation
- **Memory:** Transfers data to/from GPU
- **Progress:** Prints every 64 iterations
- **Output:** Iteration matrix on CPU
- **Fallback:** Uses CPU if GPU unavailable

##### `mandelbrot_cpu(xmin, xmax, ymin, ymax)`
CPU fallback implementation.
- **Same interface as GPU version**
- **Used when:** CuPy not installed or CUDA unavailable

##### `mandelbrot_kernel_gpu(c_real, c_imag, max_iter, width, height, output)`
CUDA kernel for parallel Mandelbrot computation.
- **Parallelization:** One thread per pixel
- **Grid/Block:** Optimized for GPU architecture
- **Output:** Writes directly to GPU array

---

### 4. make_animation_advanced.py (Animation Generator)

**Purpose:** Standalone script for creating zoom animations

**Main Function:** `create_zoom_animation()`

**Parameters:**
- `width, height: int` - Frame dimensions (default: 600×450)
- `max_iter: int` - Maximum iterations (default: 5000)
- `frames: int` - Number of frames (default: 150)
- `target_x, target_y: float` - Zoom target coordinates
- `initial_width, final_width: float` - Zoom range

**Process:**
1. Generate logarithmic zoom scales
2. For each frame:
   - Calculate bounds
   - Render with GPU
   - Apply advanced effects
   - Convert to uint8
3. Compile frames into GIF with imageio

**Output:** `gifs/zoom_advanced.gif` (150 frames @ 8 fps)

---

### 5. advanced_mandelbrot.py (Advanced Effects)

**Purpose:** Advanced rendering algorithms and effects

**Functions:**

##### `sin_colortable_advanced(rgb_thetas, ncol)`
Generate sinusoidal color palette.

##### `blinn_phong(normal, light)`
3D lighting calculation.

##### `smooth_iter(c, maxiter, stripe_s, stripe_sig)`
Smooth iteration with stripe averaging.

##### `color_pixel_advanced(...)`
Apply all advanced effects to a pixel.

##### `compute_set_advanced(...)`
Full advanced rendering pipeline.

---

## Data Structures

### Coordinate Preset Format (JSON)

```json
{
  "preset_name": {
    "coords": [xmin, xmax, ymin, ymax],
    "max_iter": 5000,
    "rgb_thetas": [r_theta, g_theta, b_theta],
    "ncycle": 32,
    "stripe_s": 0-32,
    "step_s": 0-100,
    "description": "Description text"
  }
}
```

**Available Presets:**
1. **crown** - Crown-like structure, deep zoom
2. **pow** - POW region with step shading
3. **octogone** - Octagonal patterns
4. **julia** - Julia-like region
5. **lightning** - Lightning formations
6. **web** - Web-like structures
7. **spiral** - Spiral patterns
8. **tiles** - Tiled regions
9. **velvet** - Seahorse valley

---

## Rendering Pipeline

### Standard Rendering Flow

```
1. Initialize Renderer (CPU or GPU)
   ↓
2. Create Coordinate Grid
   - xmin, xmax, ymin, ymax
   - width × height resolution
   ↓
3. Compute Mandelbrot Set
   - For each pixel (x, y):
     - c = complex(x, y)
     - Iterate: z = z² + c
     - Count iterations until |z| > 2
   ↓
4. Apply Color Scheme
   - Map iteration count to color
   - Use sinusoidal color table
   ↓
5. Output Image
   - Convert to uint8
   - Save as PNG
```

### Advanced Rendering Flow

```
1. Initialize Parameters
   - RGB thetas, ncycle
   - Stripe/step frequencies
   - Lighting parameters
   ↓
2. Generate Color Table
   - sin_colortable_advanced()
   - 4096 colors
   ↓
3. Compute with Effects
   - smooth_iter() for each pixel
   - Returns: iteration, stripe_avg, DEM, normal
   ↓
4. Apply Advanced Coloring
   - color_pixel_advanced()
   - Stripe shading overlay
   - Step shading overlay
   - Blinn-Phong lighting
   - DEM (Distance Estimation)
   ↓
5. Output High-Quality Image
```

---

## Performance Metrics

### Typical Performance (800×600, CUDA GPU)

| Iterations | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 256       | ~1000ms  | ~400ms   | 2.5x    |
| 512       | ~2000ms  | ~750ms   | 2.7x    |
| 1024      | ~4000ms  | ~1450ms  | 2.8x    |
| 2048      | ~7500ms  | ~3150ms  | 2.4x    |

### Memory Usage

- **800×600:** ~7.32 MB
- **1920×1080:** ~24.88 MB
- **2560×1440:** ~44.16 MB

### Throughput

- **CPU:** 0.06 - 0.48 M pixels/sec
- **GPU:** 0.15 - 1.2 M pixels/sec

---

## User Workflows

### Workflow 1: Quick Comparison

```
1. Run: python compare_cpu_gpu.py
2. Select: Option 1 (Side-by-side comparison)
3. Choose: Resolution (1 = 800×600, 2 = 1920×1080, 3 = 2560×1440)
4. Wait: ~30 seconds for all 4 comparisons
5. View: 4 windows with comparison images
6. Check: images/[resolution]/comparison_*.png files saved
```

### Workflow 2: Interactive Exploration

```
1. Run: python compare_cpu_gpu.py
2. Select: Option 2 (Dual interactive explorer)
3. Explore: Click to zoom in/out
4. Compare: See CPU vs GPU times in real-time
5. Close: Window to trigger analysis
6. View: Performance analysis graphs
```

### Workflow 3: Custom Render

```
1. Run: python compare_cpu_gpu.py
2. Select: Option 5 (Render custom coordinates)
3. Choose: Option 1 (Use preset)
4. Enter: Preset name (e.g., "crown")
5. Optionally: Change output filename
6. Wait: Rendering completes
7. View: images/[filename].png
```

### Workflow 4: Create Animation

```
1. Run: python compare_cpu_gpu.py
2. Select: Option 4 (Create zoom animation)
3. Configure: RGB thetas, effects, lighting
4. Enter: Output filename
5. Wait: ~2-5 minutes for 150 frames
6. View: gifs/[filename].gif
```

---

## Technical Implementation

### GPU Acceleration Strategy

**CuPy Integration:**
- Automatic GPU detection
- Graceful CPU fallback
- Memory management
- Kernel optimization

**CUDA Kernel Design:**
```python
# Grid: (blocks_x, blocks_y)
# Block: (threads_x, threads_y)
# Total threads = width × height
# Each thread computes one pixel
```

### Numba JIT Compilation

**Benefits:**
- Near C-speed performance
- No manual compilation
- Type inference
- Loop optimization

**Usage:**
```python
@jit
def mandelbrot_kernel(c, max_iter):
    # Compiled to machine code
    # ~10-100x speedup over pure Python
```

### Color Mapping Algorithm

**Sinusoidal Color Table:**
```python
R = 0.5 + 0.5 * sin(2π(phase + θ_r))
G = 0.5 + 0.5 * sin(2π(phase + θ_g))
B = 0.5 + 0.5 * sin(2π(phase + θ_b))
```

**Smooth Coloring:**
```python
smooth_i = 1 - log(log(|z|)) / log(2)
color_index = (n + smooth_i) % ncycle
```

### Distance Estimation Method (DEM)

**Formula:**
```
DEM = |z| * log(|z|) / |dz| / 2
```

**Purpose:**
- Edge detection
- 3D-like appearance
- Depth perception

### Blinn-Phong Lighting

**Components:**
1. **Diffuse:** `L_diff = max(0, N · L)`
2. **Specular:** `L_spec = max(0, N · H)^shininess`
3. **Total:** `brightness = ambient + diffuse + specular`

**Parameters:**
- Azimuth: Light horizontal angle
- Elevation: Light vertical angle
- Shininess: Specular exponent (default: 20)

---

## Output Specifications

### Comparison Images
Saved in resolution-specific subfolders:
- **Location:** `images/[resolution]/comparison_[iterations]_iterations.png`
  - `images/800x600/` - Standard resolution
  - `images/1920x1080/` - Full HD
  - `images/2560x1440/` - QHD/2K
- **Format:** PNG, 24-bit RGB
- **DPI:** 150
- **File Size:** ~230-240 KB each
- **Layout:** 2×2 grid (images + vertical summary)
- **Dynamic Sizing:** Figure dimensions and fonts adjust per resolution
- **No Overlapping:** Content fits perfectly within each resolution

### Custom Renders
- **Resolution:** 1920 × 1080 pixels
- **Format:** PNG, 24-bit RGB
- **Location:** `images/` directory
- **Effects:** Optional advanced rendering

### Animations
- **Resolution:** 600 × 450 pixels (configurable)
- **Format:** GIF
- **Frames:** 150 (default)
- **FPS:** 8
- **Duration:** ~18.75 seconds
- **File Size:** Varies (typically 5-20 MB)
- **Location:** `gifs/` directory

### Analysis Graphs
- **Resolution:** 2700 × 1500 pixels
- **Format:** PNG
- **Layout:** 2×2 panel grid
- **File:** `interactive_performance_analysis.png`

---

## Conclusion

This Mandelbrot set renderer represents a comprehensive solution for fractal visualization, combining:

1. **Performance:** GPU acceleration for 2-3x speedup
2. **Usability:** Menu-driven interface with 6 modes
3. **Quality:** Advanced rendering effects and high resolution
4. **Analysis:** Detailed performance metrics and visualization
5. **Flexibility:** Preset system and custom coordinates
6. **Extensibility:** Modular architecture for easy enhancement

The project successfully demonstrates the power of GPU computing for computationally intensive tasks while maintaining accessibility through CPU fallback and user-friendly interfaces.

---

**Project Statistics:**
- **Total Lines of Code:** ~3,500+
- **Core Modules:** 5
- **Functions:** 30+
- **Classes:** 7
- **Presets:** 9
- **Output Types:** 4 (comparison, custom, animation, analysis)
- **Supported Resolutions:** 3 (800×600, 1920×1080, 2560×1440)

**Last Updated:** February 2026
