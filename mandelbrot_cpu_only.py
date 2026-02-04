"""
CPU-Only Mandelbrot Renderer
For systems without GPU/CuPy support
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass


@dataclass
class ComputationMetrics:
    """Store computational performance metrics"""
    time_taken: float
    iterations_computed: int
    pixels_processed: int
    throughput: float
    memory_used: float
    device: str
    
    def __str__(self):
        return (f"{self.device} Performance:\n"
                f"  Time: {self.time_taken*1000:.2f} ms\n"
                f"  Pixels: {self.pixels_processed:,}\n"
                f"  Throughput: {self.throughput/1e6:.2f} M pixels/sec\n"
                f"  Memory: {self.memory_used:.2f} MB")


class MandelbrotCPU:
    """Pure NumPy implementation of Mandelbrot"""
    
    def __init__(self, width=800, height=600, max_iter=256):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.metrics = None
    
    def compute(self, xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25):
        """Compute Mandelbrot set"""
        print(f"\n{'='*60}")
        print("CPU COMPUTATION")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create coordinate grid
        x = np.linspace(xmin, xmax, self.width)
        y = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Complex grid
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)
        
        print(f"Grid: {self.width}×{self.height} = {self.width*self.height:,} pixels")
        print(f"Memory: {C.nbytes / 1024**2:.2f} MB")
        print("\nIterating...")
        
        iteration_count = 0
        
        # Main iteration loop
        for i in range(self.max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i
            iteration_count += np.sum(mask)
            
            if i % 32 == 0:
                escaped = np.sum(~mask)
                print(f"  Iteration {i}/{self.max_iter} - "
                      f"{escaped:,} escaped ({100*escaped/M.size:.1f}%)")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Store metrics
        self.metrics = ComputationMetrics(
            time_taken=elapsed,
            iterations_computed=iteration_count,
            pixels_processed=self.width * self.height,
            throughput=(self.width * self.height) / elapsed,
            memory_used=C.nbytes / 1024**2,
            device="CPU"
        )
        
        print(f"\n{self.metrics}")
        print(f"Total iterations: {iteration_count:,}")
        print(f"{'='*60}\n")
        
        return M
    
    def render(self, xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25):
        """Compute and display Mandelbrot"""
        M = self.compute(xmin, xmax, ymin, ymax)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(M, extent=[xmin, xmax, ymin, ymax],
                      cmap='hot', origin='lower', interpolation='bilinear')
        
        ax.set_title(f'Mandelbrot Set (CPU)\n'
                    f'Time: {self.metrics.time_taken*1000:.1f} ms | '
                    f'{self.width}×{self.height} @ {self.max_iter} iterations',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.colorbar(im, ax=ax, label='Iterations')
        
        plt.tight_layout()
        plt.show()
        
        return M


class InteractiveMandelbrotCPU:
    """Interactive Mandelbrot explorer with click-to-zoom"""
    
    def __init__(self, width=800, height=600, max_iter=256):
        self.renderer = MandelbrotCPU(width, height, max_iter)
        self.xmin = -2.5
        self.xmax = 1.0
        self.ymin = -1.25
        self.ymax = 1.25
        
        self.fig = None
        self.ax = None
        self.im = None
        
    def render(self):
        """Render current view"""
        M = self.renderer.compute(self.xmin, self.xmax, self.ymin, self.ymax)
        
        if self.fig is None:
            # First render
            self.fig, self.ax = plt.subplots(figsize=(12, 9))
            self.im = self.ax.imshow(M, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                                     cmap='hot', origin='lower', interpolation='bilinear')
            plt.colorbar(self.im, ax=self.ax, label='Iterations')
            
            # Connect event handlers
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            
            self.ax.set_xlabel('Real axis')
            self.ax.set_ylabel('Imaginary axis')
        else:
            # Update existing plot
            self.im.set_data(M)
            self.im.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        
        # Update title
        self.ax.set_title(
            f'Interactive Mandelbrot (CPU)\n'
            f'Left-click or scroll up: zoom in | Right-click or scroll down: zoom out\n'
            f'Time: {self.renderer.metrics.time_taken*1000:.1f} ms | '
            f'{self.renderer.width}×{self.renderer.height} @ {self.renderer.max_iter} iterations',
            fontsize=12, fontweight='bold'
        )
        
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        
        plt.draw()
    
    def zoom_at(self, x, y, zoom_factor):
        """Zoom in/out at specified point"""
        # Calculate new ranges
        x_range = (self.xmax - self.xmin) * zoom_factor / 2
        y_range = (self.ymax - self.ymin) * zoom_factor / 2
        
        self.xmin = x - x_range
        self.xmax = x + x_range
        self.ymin = y - y_range
        self.ymax = y + y_range
        
        self.render()
    
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if event.button == 1:  # Left click - zoom in
            self.zoom_at(x, y, 0.5)
        elif event.button == 3:  # Right click - zoom out
            self.zoom_at(x, y, 2.0)
    
    def on_scroll(self, event):
        """Handle scroll events"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if event.button == 'up':  # Scroll up - zoom in
            self.zoom_at(x, y, 0.75)
        elif event.button == 'down':  # Scroll down - zoom out
            self.zoom_at(x, y, 1.33)
    
    def explore(self):
        """Start interactive exploration"""
        print("\n" + "="*60)
        print("  INTERACTIVE MANDELBROT EXPLORER (CPU)")
        print("="*60)
        print("\nControls:")
        print("  • Left-click or scroll up: Zoom in")
        print("  • Right-click or scroll down: Zoom out")
        print("  • Close window to exit")
        print("="*60 + "\n")
        
        self.render()
        plt.show()


def main():
    """Demo CPU-only version"""
    print("\n" + "="*60)
    print("  MANDELBROT SET - CPU Version")
    print("  Pure NumPy Implementation")
    print("="*60)
    print("\n1. Static render")
    print("2. Interactive explorer")
    
    choice = input("\nChoose mode (1/2, default=2): ").strip() or "2"
    
    if choice == "1":
        mandelbrot = MandelbrotCPU(width=800, height=600, max_iter=256)
        mandelbrot.render()
        print("\n[OK] Rendering complete!")
    else:
        explorer = InteractiveMandelbrotCPU(width=800, height=600, max_iter=256)
        explorer.explore()
    
    print("\nFor GPU acceleration, install CuPy:")
    print("  pip install cupy-cuda11x  (or cupy-cuda12x)")
    print("  Then run: python mandelbrot_gpu.py")


if __name__ == "__main__":
    main()
