# Mandelbrot Set: CPU vs GPU Comparison Tool

A high-performance Mandelbrot set renderer with CPU/GPU comparison, interactive exploration, and advanced visualization effects.

## 🚀 Features

- **Side-by-Side CPU vs GPU Comparison** - Compare rendering performance across multiple iteration counts
- **Dual Interactive Explorer** - Synchronized CPU and GPU explorers with real-time zoom/pan
- **Performance Analysis** - Detailed metrics, graphs, and speedup calculations
- **Advanced Rendering** - Stripe shading, step shading, and Blinn-Phong lighting effects
- **Zoom Animations** - Create stunning GIF animations with advanced effects
- **Custom Coordinates** - Render specific regions with preset support
- **High Resolution Output** - Support for 800×600, 1920×1080, and 2560×1440 resolutions

## 📋 Requirements

### Core Dependencies
```
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=9.0.0
```

### GPU Acceleration (Optional but Recommended)
For CUDA 11.x:
```bash
pip install cupy-cuda11x>=12.0.0
```

For CUDA 12.x:
```bash
pip install cupy-cuda12x>=12.0.0
```

Check your CUDA version:
```bash
nvcc --version
# or
nvidia-smi
```

### Installation
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

Run the main comparison tool:
```bash
python compare_cpu_gpu.py
```

You'll see a menu with 7 options:

### Option 1: Side-by-Side Comparison
Compares CPU vs GPU rendering at 256, 512, 1024, and 2048 iterations.
- Choose resolution (800×600, 1920×1080, or 2560×1440)
- Images saved in resolution-specific subfolders (e.g., `images/800x600/`)
- Layout automatically adjusts to fit each resolution perfectly
- Displays vertical comparison summary below images without overlapping

### Option 2: Dual Interactive Explorer
Launch synchronized CPU and GPU explorers:
- Fixed 800×600 resolution for optimal interactive performance
- Left-click to zoom in
- Right-click to zoom out
- Both views stay synchronized
- Metrics collected for each interaction
- Automatically opens analysis when closed

### Option 3: Comparative Analysis
View performance graphs and statistics:
- Render time per interaction
- GPU speedup per interaction
- Cumulative time comparison
- Summary statistics
- Requires data from Option 2

### Option 4: Create Zoom Animation
Generate advanced zoom animations:
- Configure RGB color schemes
- Adjust stripe and step shading
- Set lighting parameters
- Output saved to `gifs/` directory
- Default: 150 frames at 8 fps

### Option 5: Render Custom Coordinates
Render specific regions:
- Use presets (crown, julia, lightning, etc.)
- Enter custom coordinates
- Save new presets
- Advanced rendering options
- Output saved to `images/` directory

### Option 6: Run All Comparisons
Executes options 1-3 sequentially

### Option 7: Exit
Close the application

## 📁 Project Structure

### Core Files

**`compare_cpu_gpu.py`** - Main comparison tool
- `ComparisonTool` class with all comparison methods
- Menu-driven interface
- Handles all 6 comparison modes

**`mandelbrot_cpu_only.py`** - CPU renderer
- `MandelbrotCPU` class for basic CPU rendering
- `InteractiveMandelbrotCPU` for interactive exploration
- Numba JIT compilation for performance

**`mandelbrot_gpu.py`** - GPU renderer
- `MandelbrotRenderer` class with GPU/CPU fallback
- CuPy-based CUDA acceleration
- Progress tracking and metrics

**`make_animation_advanced.py`** - Advanced animation script
- Standalone animation generator
- Advanced rendering effects
- Configurable parameters

**`advanced_mandelbrot.py`** - Advanced rendering functions
- Stripe shading algorithms
- Step shading effects
- Blinn-Phong lighting
- Custom color tables

### Data Files

**`coordinate_presets.json`** - Saved coordinate presets
- 9 pre-configured interesting regions
- Custom RGB color schemes
- Advanced rendering parameters
- Presets: crown, pow, octogone, julia, lightning, web, spiral, tiles, velvet

**`requirements.txt`** - Python dependencies

### Output Directories

**`images/`** - Rendered images organized by resolution
- `images/800x600/` - Standard resolution comparisons
- `images/1920x1080/` - Full HD comparisons
- `images/2560x1440/` - QHD/2K comparisons
- Custom coordinate renders

**`gifs/`** - Generated zoom animations

## 🎨 Advanced Rendering Effects

### RGB Thetas
Control color scheme with 3 phase offsets (0-1):
- `(0.0, 0.15, 0.25)` - Blue-green palette
- `(0.85, 0.0, 0.15)` - Vibrant colors
- `(0.5, 0.7, 0.9)` - Warm tones

### Stripe Shading
Adds organic texture patterns (0-32):
- `0` - Disabled
- `8` - Moderate stripes
- `16+` - Dense stripes

### Step Shading
Creates contour-like bands (0-100):
- `0` - Disabled
- `10` - Subtle contours
- `20+` - Pronounced steps

### Lighting
Blinn-Phong lighting for 3D appearance:
- Azimuth: 0-360° (horizontal angle)
- Elevation: 0-90° (vertical angle)
- Default: 45°, 45°

## 📊 Output Files

### Comparison Images
Saved in resolution-specific subfolders:
- `images/800x600/comparison_256_iterations.png`
- `images/800x600/comparison_512_iterations.png`
- `images/800x600/comparison_1024_iterations.png`
- `images/800x600/comparison_2048_iterations.png`

(Similar structure for 1920x1080 and 2560x1440)

**Features:**
- Layout optimized for each resolution
- Vertical comparison summary
- No overlapping content
- Format: PNG

### Performance Analysis
- `interactive_performance_analysis.png` - 4-panel analysis graph
  - Render time per interaction
  - GPU speedup chart
  - Cumulative time comparison
  - Summary statistics

### Custom Renders
- Saved in `images/` directory
- Named by preset or custom filename
- Resolution: 1920×1080 pixels
- Format: PNG

### Animations
- Saved in `gifs/` directory
- Default: `zoom_advanced.gif`
- 150 frames at 8 fps
- ~18.75 seconds duration

## 🔧 Performance Tips

1. **GPU Acceleration**: Install CuPy for 2-3x speedup
2. **Resolution**: Lower resolution for faster rendering
3. **Iterations**: Fewer iterations = faster computation
4. **Advanced Effects**: Disable for maximum speed

## 📈 Typical Performance

With CUDA GPU (800×600 resolution):
- 256 iterations: ~1-2 seconds
- 512 iterations: ~2-3 seconds
- 1024 iterations: ~3-5 seconds
- 2048 iterations: ~6-8 seconds

GPU speedup: 2-3x faster than CPU

## 🐛 Troubleshooting

**No GPU detected**: CuPy not installed or CUDA unavailable
- Tool will automatically fall back to CPU
- Install CuPy matching your CUDA version

**Import errors**: Missing dependencies
```bash
pip install -r requirements.txt
```

**Slow rendering**: Normal for CPU-only mode
- Consider installing GPU support
- Reduce resolution or iterations

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests.

## 📧 Support

For questions or issues, please create an issue in the repository.

---

**Enjoy exploring the infinite beauty of the Mandelbrot set! 🌀**
