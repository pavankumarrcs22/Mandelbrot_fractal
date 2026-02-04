"""
GPU-Accelerated Mandelbrot Set Renderer with Interactive Visualization
Similar to: https://github.com/jlesuffleur/gpu_mandelbrot

Features:
- CPU and GPU (CUDA) implementations
- Real-time performance comparison
- Interactive visualization with zoom
- Detailed computational metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from dataclasses import dataclass
from typing import Tuple

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] CuPy found - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - GPU acceleration disabled")
    print("  Install with: pip install cupy-cuda11x  (or cupy-cuda12x)")


@dataclass
class ComputationMetrics:
    """Store computational performance metrics"""
    time_taken: float
    iterations_computed: int
    pixels_processed: int
    throughput: float  # pixels/second
    memory_used: float  # MB
    device: str
    
    def __str__(self):
        return (f"{self.device} Performance:\n"
                f"  Time: {self.time_taken*1000:.2f} ms\n"
                f"  Pixels: {self.pixels_processed:,}\n"
                f"  Throughput: {self.throughput/1e6:.2f} M pixels/sec\n"
                f"  Memory: {self.memory_used:.2f} MB")


class MandelbrotRenderer:
    """Mandelbrot set renderer with CPU and GPU support"""
    
    def __init__(self, width=800, height=600, max_iter=256):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        
        # Default view window
        self.xmin, self.xmax = -2.5, 1.0
        self.ymin, self.ymax = -1.25, 1.25
        
        # Performance tracking
        self.cpu_metrics = None
        self.gpu_metrics = None
        
    def mandelbrot_cpu(self, xmin, xmax, ymin, ymax):
        """CPU implementation of Mandelbrot set"""
        print(f"\n{'='*60}")
        print("CPU COMPUTATION")
        print(f"{'='*60}")
        
        # Create coordinate arrays
        start_time = time.time()
        
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Complex number grid
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)
        
        print(f"Grid created: {self.width}x{self.height} = {self.width*self.height:,} pixels")
        print(f"Memory allocated: {C.nbytes / 1024**2:.2f} MB")
        print("\nIterating...")
        
        iteration_count = 0
        # Mandelbrot iteration
        for i in range(self.max_iter):
            # Compute z = z^2 + c for all points
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i
            iteration_count += np.sum(mask)
            
            # Progress indicator
            if i % 32 == 0:
                print(f"  Iteration {i}/{self.max_iter} - "
                      f"{np.sum(~mask):,} pixels escaped "
                      f"({100*np.sum(~mask)/M.size:.1f}%)")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        self.cpu_metrics = ComputationMetrics(
            time_taken=elapsed,
            iterations_computed=iteration_count,
            pixels_processed=self.width * self.height,
            throughput=(self.width * self.height) / elapsed,
            memory_used=C.nbytes / 1024**2,
            device="CPU"
        )
        
        print(f"\n{self.cpu_metrics}")
        print(f"Total iterations computed: {iteration_count:,}")
        print(f"{'='*60}\n")
        
        return M
    
    def mandelbrot_gpu(self, xmin, xmax, ymin, ymax):
        """GPU implementation using CuPy"""
        if not GPU_AVAILABLE:
            print("GPU not available. Using CPU instead.")
            return self.mandelbrot_cpu(xmin, xmax, ymin, ymax)
        
        print(f"\n{'='*60}")
        print("GPU COMPUTATION")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create coordinate arrays on GPU
        x = cp.linspace(xmin, xmax, self.width)
        y = cp.linspace(ymin, ymax, self.height)
        X, Y = cp.meshgrid(x, y)
        
        # Complex number grid on GPU
        C = X + 1j * Y
        Z = cp.zeros_like(C)
        M = cp.zeros(C.shape, dtype=cp.int32)
        
        print(f"Grid created on GPU: {self.width}x{self.height} = {self.width*self.height:,} pixels")
        print(f"GPU memory allocated: {C.nbytes / 1024**2:.2f} MB")
        print("\nIterating on GPU...")
        
        iteration_count = 0
        # Mandelbrot iteration on GPU
        for i in range(self.max_iter):
            mask = cp.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i
            iteration_count += int(cp.sum(mask))
            
            # Progress indicator (less frequent to reduce GPU sync overhead)
            if i % 64 == 0:
                escaped = int(cp.sum(~mask))
                print(f"  Iteration {i}/{self.max_iter} - "
                      f"{escaped:,} pixels escaped "
                      f"({100*escaped/M.size:.1f}%)")
        
        # Transfer result back to CPU
        M_cpu = cp.asnumpy(M)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        self.gpu_metrics = ComputationMetrics(
            time_taken=elapsed,
            iterations_computed=iteration_count,
            pixels_processed=self.width * self.height,
            throughput=(self.width * self.height) / elapsed,
            memory_used=C.nbytes / 1024**2,
            device="GPU (CUDA)"
        )
        
        print(f"\n{self.gpu_metrics}")
        print(f"Total iterations computed: {iteration_count:,}")
        print(f"{'='*60}\n")
        
        return M_cpu
    
    def compare_performance(self):
        """Display side-by-side performance comparison"""
        if self.cpu_metrics is None or self.gpu_metrics is None:
            print("Run both CPU and GPU computations first!")
            return
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}\n")
        
        # Time comparison
        speedup = self.cpu_metrics.time_taken / self.gpu_metrics.time_taken
        print(f"Time Comparison:")
        print(f"  CPU:  {self.cpu_metrics.time_taken*1000:.2f} ms")
        print(f"  GPU:  {self.gpu_metrics.time_taken*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x faster on GPU")
        print()
        
        # Throughput comparison
        print(f"Throughput Comparison:")
        print(f"  CPU:  {self.cpu_metrics.throughput/1e6:.2f} M pixels/sec")
        print(f"  GPU:  {self.gpu_metrics.throughput/1e6:.2f} M pixels/sec")
        print(f"  Improvement: {self.gpu_metrics.throughput/self.cpu_metrics.throughput:.1f}x")
        print()
        
        # Iteration comparison
        print(f"Computational Work:")
        print(f"  CPU iterations: {self.cpu_metrics.iterations_computed:,}")
        print(f"  GPU iterations: {self.gpu_metrics.iterations_computed:,}")
        print()
        
        # Memory usage
        print(f"Memory Usage:")
        print(f"  CPU:  {self.cpu_metrics.memory_used:.2f} MB")
        print(f"  GPU:  {self.gpu_metrics.memory_used:.2f} MB")
        print()
        
        # Efficiency metrics
        print(f"Computational Efficiency:")
        cpu_iter_per_sec = self.cpu_metrics.iterations_computed / self.cpu_metrics.time_taken
        gpu_iter_per_sec = self.gpu_metrics.iterations_computed / self.gpu_metrics.time_taken
        print(f"  CPU:  {cpu_iter_per_sec/1e6:.2f} M iterations/sec")
        print(f"  GPU:  {gpu_iter_per_sec/1e6:.2f} M iterations/sec")
        print(f"  GPU advantage: {gpu_iter_per_sec/cpu_iter_per_sec:.1f}x")
        
        print(f"\n{'='*60}\n")
        
        return speedup
    
    def render_comparison(self):
        """Render and display both CPU and GPU results side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        print("\n" + "="*60)
        print("RENDERING MANDELBROT SET")
        print("="*60)
        
        # CPU computation
        M_cpu = self.mandelbrot_cpu(self.xmin, self.xmax, self.ymin, self.ymax)
        
        # GPU computation
        M_gpu = self.mandelbrot_gpu(self.xmin, self.xmax, self.ymin, self.ymax)
        
        # Display comparison
        speedup = self.compare_performance()
        
        # Plot CPU result
        im1 = axes[0].imshow(M_cpu, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                            cmap='hot', origin='lower', interpolation='bilinear')
        axes[0].set_title(f'CPU Result\n{self.cpu_metrics.time_taken*1000:.1f} ms', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Real axis')
        axes[0].set_ylabel('Imaginary axis')
        plt.colorbar(im1, ax=axes[0], label='Iterations')
        
        # Plot GPU result
        im2 = axes[1].imshow(M_gpu, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                            cmap='hot', origin='lower', interpolation='bilinear')
        axes[1].set_title(f'GPU Result\n{self.gpu_metrics.time_taken*1000:.1f} ms ({speedup:.1f}x faster)', 
                         fontsize=14, fontweight='bold', color='green')
        axes[1].set_xlabel('Real axis')
        axes[1].set_ylabel('Imaginary axis')
        plt.colorbar(im2, ax=axes[1], label='Iterations')
        
        plt.suptitle(f'Mandelbrot Set - {self.width}x{self.height} @ {self.max_iter} iterations',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, M_cpu, M_gpu


class InteractiveMandelbrot:
    """Interactive Mandelbrot explorer with zoom capability"""
    
    def __init__(self, width=800, height=600, max_iter=256, use_gpu=True):
        self.renderer = MandelbrotRenderer(width, height, max_iter)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.zoom_stack = []
        
        print(f"\n{'='*60}")
        print("INTERACTIVE MANDELBROT EXPLORER")
        print(f"{'='*60}")
        print(f"Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        print(f"Resolution: {width}x{height}")
        print(f"Max iterations: {max_iter}")
        print(f"{'='*60}\n")
    
    def render(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """Render Mandelbrot set in specified region"""
        if xmin is not None:
            self.renderer.xmin = xmin
            self.renderer.xmax = xmax
            self.renderer.ymin = ymin
            self.renderer.ymax = ymax
        
        # Choose rendering method
        if self.use_gpu:
            M = self.renderer.mandelbrot_gpu(
                self.renderer.xmin, self.renderer.xmax,
                self.renderer.ymin, self.renderer.ymax
            )
            metrics = self.renderer.gpu_metrics
        else:
            M = self.renderer.mandelbrot_cpu(
                self.renderer.xmin, self.renderer.xmax,
                self.renderer.ymin, self.renderer.ymax
            )
            metrics = self.renderer.cpu_metrics
        
        # Display
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(M, extent=[self.renderer.xmin, self.renderer.xmax, 
                                   self.renderer.ymin, self.renderer.ymax],
                      cmap='hot', origin='lower', interpolation='bilinear')
        
        device = "GPU" if self.use_gpu else "CPU"
        ax.set_title(f'Mandelbrot Set - {device}\n'
                    f'Time: {metrics.time_taken*1000:.1f} ms | '
                    f'Throughput: {metrics.throughput/1e6:.1f} M pixels/sec',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.colorbar(im, ax=ax, label='Iterations to escape')
        
        # Add zoom instructions
        ax.text(0.02, 0.98, 'Click to zoom in\nRight-click to zoom out',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Connect zoom event
        fig.canvas.mpl_connect('button_press_event', 
                              lambda event: self.on_click(event, ax, fig))
        
        plt.tight_layout()
        plt.show()
        
        return M
    
    def on_click(self, event, ax, fig):
        """Handle zoom on click"""
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # Left click - zoom in
            # Save current view
            self.zoom_stack.append((
                self.renderer.xmin, self.renderer.xmax,
                self.renderer.ymin, self.renderer.ymax
            ))
            
            # Calculate new bounds (zoom 4x around click point)
            x_center = event.xdata
            y_center = event.ydata
            x_range = (self.renderer.xmax - self.renderer.xmin) / 4
            y_range = (self.renderer.ymax - self.renderer.ymin) / 4
            
            plt.close(fig)
            self.render(
                x_center - x_range, x_center + x_range,
                y_center - y_range, y_center + y_range
            )
            
        elif event.button == 3 and self.zoom_stack:  # Right click - zoom out
            xmin, xmax, ymin, ymax = self.zoom_stack.pop()
            plt.close(fig)
            self.render(xmin, xmax, ymin, ymax)


def demo_performance_scaling():
    """Demonstrate how performance scales with problem size"""
    print("\n" + "="*60)
    print("PERFORMANCE SCALING ANALYSIS")
    print("="*60 + "\n")
    
    resolutions = [
        (400, 300),
        (800, 600),
        (1200, 900),
        (1600, 1200),
    ]
    
    results = []
    
    for width, height in resolutions:
        print(f"\nTesting {width}x{height}...")
        renderer = MandelbrotRenderer(width, height, max_iter=256)
        
        # CPU test
        M_cpu = renderer.mandelbrot_cpu(-2.5, 1.0, -1.25, 1.25)
        
        # GPU test (if available)
        if GPU_AVAILABLE:
            M_gpu = renderer.mandelbrot_gpu(-2.5, 1.0, -1.25, 1.25)
            speedup = renderer.cpu_metrics.time_taken / renderer.gpu_metrics.time_taken
        else:
            speedup = None
        
        results.append({
            'resolution': f'{width}x{height}',
            'pixels': width * height,
            'cpu_time': renderer.cpu_metrics.time_taken,
            'gpu_time': renderer.gpu_metrics.time_taken if GPU_AVAILABLE else None,
            'speedup': speedup
        })
    
    # Display results table
    print("\n" + "="*60)
    print("SCALING RESULTS")
    print("="*60)
    print(f"{'Resolution':<12} {'Pixels':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-"*60)
    
    for r in results:
        gpu_time_str = f"{r['gpu_time']*1000:.1f}" if r['gpu_time'] else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        print(f"{r['resolution']:<12} {r['pixels']:<12,} "
              f"{r['cpu_time']*1000:<12.1f} {gpu_time_str:<12} {speedup_str:<10}")
    
    print("="*60 + "\n")
    
    # Plot scaling
    if GPU_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        pixels = [r['pixels'] for r in results]
        cpu_times = [r['cpu_time']*1000 for r in results]
        gpu_times = [r['gpu_time']*1000 for r in results]
        speedups = [r['speedup'] for r in results]
        
        # Time comparison
        ax1.plot(pixels, cpu_times, 'o-', label='CPU', linewidth=2, markersize=8)
        ax1.plot(pixels, gpu_times, 's-', label='GPU', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Pixels')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Computation Time vs Problem Size', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup
        ax2.plot(pixels, speedups, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Pixels')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('GPU Speedup vs Problem Size', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('performance_scaling.png', dpi=150, bbox_inches='tight')
        print("Saved scaling plot to: performance_scaling.png\n")
        plt.show()


def main():
    """Main demonstration"""
    print("\n" + "="*60)
    print("  MANDELBROT SET - GPU vs CPU")
    print("  Python Implementation with CuPy")
    print("="*60)
    
    # Check GPU availability
    if GPU_AVAILABLE:
        print("\n[OK] GPU acceleration available")
        try:
            print(f"GPU: {cp.cuda.Device().name}")
            mem_info = cp.cuda.Device().mem_info
            print(f"GPU Memory: {mem_info[1]/1024**3:.1f} GB total, "
                  f"{mem_info[0]/1024**3:.1f} GB free")
        except:
            pass
    else:
        print("\n✗ GPU not available - install CuPy for GPU acceleration")
    
    print("\n" + "="*60)
    print("DEMO OPTIONS")
    print("="*60)
    print("1. Side-by-side CPU vs GPU comparison")
    print("2. Interactive explorer (GPU if available)")
    print("3. Performance scaling analysis")
    print("4. All of the above")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1' or choice == '4':
        # Comparison demo
        renderer = MandelbrotRenderer(width=800, height=600, max_iter=256)
        fig, M_cpu, M_gpu = renderer.render_comparison()
        plt.savefig('mandelbrot_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved comparison to: mandelbrot_comparison.png")
        plt.show()
    
    if choice == '2' or choice == '4':
        # Interactive explorer
        explorer = InteractiveMandelbrot(width=1000, height=800, max_iter=256)
        explorer.render()
    
    if choice == '3' or choice == '4':
        # Scaling analysis
        demo_performance_scaling()
    
    print("\n[OK] Demo complete!")


if __name__ == "__main__":
    main()
