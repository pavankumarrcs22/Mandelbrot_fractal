"""
Unified CPU vs GPU Comparison Tool
Provides side-by-side comparisons, dual interactive explorers, and performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import math
import os
from dataclasses import dataclass, field
from typing import List
from numba import jit

# Import CPU renderer
from mandelbrot_cpu_only import MandelbrotCPU, InteractiveMandelbrotCPU

# Import GPU renderer
from mandelbrot_gpu import MandelbrotRenderer, GPU_AVAILABLE


# Advanced rendering functions
def sin_colortable_advanced(rgb_thetas=(0.0, 0.15, 0.25), ncol=2**12):
    """Create sinusoidal color table for advanced rendering"""
    def colormap(x, rgb_thetas):
        y = np.column_stack(((x + rgb_thetas[0]) * 2 * math.pi,
                             (x + rgb_thetas[1]) * 2 * math.pi,
                             (x + rgb_thetas[2]) * 2 * math.pi))
        val = 0.5 + 0.5 * np.sin(y)
        return val
    return colormap(np.linspace(0, 1, ncol), rgb_thetas)


@jit
def blinn_phong(normal, light):
    """Blinn-Phong shading algorithm"""
    normal = normal / abs(normal)
    
    # Diffuse light
    ldiff = (normal.real * math.cos(light[0]) * math.cos(light[1]) +
             normal.imag * math.sin(light[0]) * math.cos(light[1]) +
             1 * math.sin(light[1]))
    ldiff = ldiff / (1 + 1 * math.sin(light[1]))
    
    # Specular light
    phi_half = (math.pi/2 + light[1]) / 2
    lspec = (normal.real * math.cos(light[0]) * math.sin(phi_half) +
             normal.imag * math.sin(light[0]) * math.sin(phi_half) +
             1 * math.cos(phi_half))
    lspec = lspec / (1 + 1 * math.cos(phi_half))
    lspec = lspec ** light[6]
    
    bright = light[3] + light[4] * ldiff + light[5] * lspec
    bright = bright * light[2] + (1 - light[2]) / 2
    return bright


@jit
def smooth_iter(c, maxiter, stripe_s, stripe_sig):
    """Compute smooth iteration count with stripe data"""
    esc_radius_2 = 10**10
    z = complex(0, 0)
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a = 0
    dz = 1 + 0j
    
    for n in range(maxiter):
        dz = dz * 2 * z + 1
        z = z * z + c
        
        if stripe:
            stripe_t = (math.sin(stripe_s * math.atan2(z.imag, z.real)) + 1) / 2
        
        if z.real * z.real + z.imag * z.imag > esc_radius_2:
            modz = abs(z)
            log_ratio = 2 * math.log(modz) / math.log(esc_radius_2)
            smooth_i = 1 - math.log(log_ratio) / math.log(2)
            
            if stripe:
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig - 1)) +
                           stripe_t * smooth_i * (1 - stripe_sig))
                stripe_a = stripe_a / (1 - stripe_sig**n *
                                      (1 + smooth_i * (stripe_sig - 1)))
            
            normal = z / dz
            dem = modz * math.log(modz) / abs(dz) / 2
            
            return (n + smooth_i, stripe_a, dem, normal)
        
        if stripe:
            stripe_a = stripe_a * stripe_sig + stripe_t * (1 - stripe_sig)
    
    return (0, 0, 0, 0)


@jit
def color_pixel_advanced(matxy, niter, stripe_a, step_s, dem, normal, colortable,
                        ncycle, light):
    """Color pixel with advanced effects"""
    ncol = colortable.shape[0] - 1
    niter = math.sqrt(niter) % ncycle / ncycle
    col_i = round(niter * ncol)
    
    def overlay(x, y, gamma):
        if (2 * y) < 1:
            out = 2 * x * y
        else:
            out = 1 - 2 * (1 - x) * (1 - y)
        return out * gamma + x * (1 - gamma)
    
    bright = blinn_phong(normal, light)
    
    dem = -math.log(dem) / 12
    dem = 1 / (1 + math.exp(-10 * ((2 * dem - 1) / 2)))
    
    nshader = 0
    shader = 0
    
    if stripe_a > 0:
        nshader += 1
        shader = shader + stripe_a
    
    if step_s > 0:
        step_s = 1 / step_s
        col_i = round((niter - niter % step_s) * ncol)
        x = niter % step_s / step_s
        light_step = 6 * (1 - x**5 - (1 - x)**100) / 10
        step_s = step_s / 8
        x = niter % step_s / step_s
        light_step2 = 6 * (1 - x**5 - (1 - x)**30) / 10
        light_step = overlay(light_step2, light_step, 1)
        nshader += 1
        shader = shader + light_step
    
    if nshader > 0:
        bright = overlay(bright, shader / nshader, 1) * (1 - dem) + dem * bright
    
    for i in range(3):
        matxy[i] = colortable[col_i, i]
        matxy[i] = overlay(matxy[i], bright, 1)
        matxy[i] = max(0, min(1, matxy[i]))


@jit
def compute_set_advanced(creal, cim, maxiter, colortable, ncycle, stripe_s, 
                        stripe_sig, step_s, diag, light):
    """Compute Mandelbrot set with advanced rendering"""
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3))
    
    for x in range(xpixels):
        for y in range(ypixels):
            c = complex(creal[x], cim[y])
            niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s, stripe_sig)
            if niter > 0:
                color_pixel_advanced(mat[y, x, ], niter, stripe_a, step_s, dem / diag,
                                   normal, colortable, ncycle, light)
    return mat


@dataclass
class ComparisonMetrics:
    """Store comparison metrics across multiple runs"""
    iteration_counts: List[int] = field(default_factory=list)
    cpu_times: List[float] = field(default_factory=list)
    gpu_times: List[float] = field(default_factory=list)
    cpu_throughputs: List[float] = field(default_factory=list)
    gpu_throughputs: List[float] = field(default_factory=list)
    speedups: List[float] = field(default_factory=list)


class ComparisonTool:
    """Unified tool for CPU vs GPU comparisons"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.images_dir = 'images'
        self.gifs_dir = 'gifs'
        
        # Create output directories if they don't exist
        for directory in [self.images_dir, self.gifs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        self.metrics = ComparisonMetrics()
        self.interactive_metrics = []  # Store metrics from interactive sessions
        
    def apply_color_scheme(self, M, max_iter, rgb_thetas=(0.85, 0.0, 0.15)):
        """Apply sinusoidal color scheme to iteration matrix
        
        Args:
            M: Iteration matrix
            max_iter: Maximum iterations
            rgb_thetas: Tuple of (r_theta, g_theta, b_theta) phase offsets for color generation
        """
        # Generate color table with custom RGB thetas
        ncycle = 32
        colortable = np.zeros((ncycle, 3), dtype=np.uint8)
        for i in range(ncycle):
            phase = i / ncycle
            colortable[i, 0] = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * (phase + rgb_thetas[0]))))
            colortable[i, 1] = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * (phase + rgb_thetas[1]))))
            colortable[i, 2] = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * (phase + rgb_thetas[2]))))
        
        # Create RGB image
        img = np.zeros((*M.shape, 3), dtype=np.uint8)
        
        # Pixels that escaped
        mask_escaped = M < max_iter
        if np.any(mask_escaped):
            col_indices = (M[mask_escaped] % ncycle).astype(int)
            img[mask_escaped] = colortable[col_indices]
        
        # Pixels in the set (black)
        img[~mask_escaped] = [0, 0, 0]
        
        return img
        
    def side_by_side_comparison(self):
        """Compare CPU vs GPU at multiple iteration counts"""
        print("\n" + "="*70)
        print("  SIDE-BY-SIDE CPU vs GPU COMPARISON")
        print("="*70)
        
        # Resolution selection menu
        print("\nSelect Resolution:")
        print("="*70)
        print("1. 800 × 600   (Standard)")
        print("2. 1920 × 1080 (Full HD)")
        print("3. 2560 × 1440 (QHD/2K)")
        print("="*70)
        
        resolution_choice = input("\nEnter your choice (1-3): ").strip()
        
        if resolution_choice == '1':
            self.width, self.height = 800, 600
        elif resolution_choice == '2':
            self.width, self.height = 1920, 1080
        elif resolution_choice == '3':
            self.width, self.height = 2560, 1440
        else:
            print("\n[WARNING] Invalid choice. Using default 800×600")
            self.width, self.height = 800, 600
        
        print(f"\n[OK] Selected resolution: {self.width}×{self.height}")
        print("\n" + "="*70)
        print("Rendering all iterations in the background...")
        print("All windows will appear together when complete.")
        print("="*70)
        
        iterations = [256, 512, 1024, 2048]
        figures = []  # Store all figures
        
        for max_iter in iterations:
            print(f"\n{'='*70}")
            print(f"  Testing with {max_iter} iterations")
            print(f"{'='*70}")
            
            # CPU computation
            print("\n[CPU]")
            cpu_renderer = MandelbrotCPU(self.width, self.height, max_iter)
            M_cpu = cpu_renderer.compute()
            cpu_time = cpu_renderer.metrics.time_taken
            cpu_throughput = cpu_renderer.metrics.throughput
            
            # GPU computation
            print("\n[GPU]")
            gpu_renderer = MandelbrotRenderer(self.width, self.height, max_iter)
            if GPU_AVAILABLE:
                M_gpu = gpu_renderer.mandelbrot_gpu(-2.5, 1.0, -1.25, 1.25)
                gpu_time = gpu_renderer.gpu_metrics.time_taken
                gpu_throughput = gpu_renderer.gpu_metrics.throughput
            else:
                M_gpu = gpu_renderer.mandelbrot_cpu(-2.5, 1.0, -1.25, 1.25)
                gpu_time = gpu_renderer.cpu_metrics.time_taken
                gpu_throughput = gpu_renderer.cpu_metrics.throughput
            
            # Store metrics
            self.metrics.iteration_counts.append(max_iter)
            self.metrics.cpu_times.append(cpu_time)
            self.metrics.gpu_times.append(gpu_time)
            self.metrics.cpu_throughputs.append(cpu_throughput)
            self.metrics.gpu_throughputs.append(gpu_throughput)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            self.metrics.speedups.append(speedup)
            
            # Create individual figure for this iteration count
            fig = plt.figure(figsize=(12.8, 7.2))
            gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.05, height_ratios=[4, 1])

            
            # Apply color scheme
            img_cpu = self.apply_color_scheme(M_cpu, max_iter)
            img_gpu = self.apply_color_scheme(M_gpu, max_iter)
            
            # Plot CPU (top left)
            ax_cpu = fig.add_subplot(gs[0, 0])
            ax_cpu.imshow(img_cpu, origin='lower')
            ax_cpu.set_title(f'CPU\n{max_iter} iterations\n{cpu_time*1000:.1f} ms', 
                           fontsize=14, fontweight='bold')
            ax_cpu.axis('off')
            
            # Plot GPU (top right)
            ax_gpu = fig.add_subplot(gs[0, 1])
            ax_gpu.imshow(img_gpu, origin='lower')
            device_label = "GPU" if GPU_AVAILABLE else "CPU"
            ax_gpu.set_title(f'{device_label}\n{max_iter} iterations\n{gpu_time*1000:.1f} ms', 
                           fontsize=14, fontweight='bold')
            ax_gpu.axis('off')
            
            # Detailed comparison text (bottom, spanning both columns)
            ax_text = fig.add_subplot(gs[1, :])
            ax_text.axis('off')
            
            comparison_text = (
                f"COMPARISON SUMMARY\n\n"
                f"Iterations: {max_iter}\n"
                f"Resolution: {self.width}×{self.height}\n"
                f"Total Pixels: {self.width*self.height:,}\n\n"
                f"CPU Time: {cpu_time*1000:.2f} ms\n"
                f"GPU Time: {gpu_time*1000:.2f} ms\n\n"
                f"CPU Throughput: {cpu_throughput/1e6:.2f} M pixels/s\n"
                f"GPU Throughput: {gpu_throughput/1e6:.2f} M pixels/s\n\n"
                f"Speedup: {speedup:.2f}x\n\n"
                f"GPU Device: {'CUDA' if GPU_AVAILABLE else 'CPU (fallback)'}"
            )
            ax_text.text(0.5, 0.5, comparison_text, fontsize=10, 
                        verticalalignment='center', horizontalalignment='center',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.8))
            
            fig.suptitle(f'CPU vs GPU Comparison - {max_iter} Iterations', 
                        fontsize=14, fontweight='bold', y=0.97)
            
            # Save individual figure
            filename = f'comparison_{max_iter}_iterations.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\n[OK] Saved: {filename}")
            
            # Store figure for later display
            figures.append(fig)
        
        
        print(f"\n{'='*70}")
        print("[OK] All rendering complete!")
        print("[OK] Displaying all windows now...")
        print(f"{'='*70}")
        
        # Print performance summary table
        print("\n" + "="*80)
        print("  PERFORMANCE SUMMARY TABLE")
        print("="*80)
        print(f"{'Iterations':<12} {'Pixels':<10} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
        print("-"*80)
        
        for i, iter_count in enumerate(self.metrics.iteration_counts):
            pixels = f"{self.width * self.height / 1000:.0f}K"
            cpu_time = f"{self.metrics.cpu_times[i]*1000:.1f}ms"
            gpu_time = f"{self.metrics.gpu_times[i]*1000:.1f}ms"
            speedup = f"{self.metrics.speedups[i]:.2f}x"
            
            print(f"{iter_count:<12} {pixels:<10} {cpu_time:<12} {gpu_time:<12} {speedup:<10}")
        
        print("="*80)
        print(f"Resolution: {self.width}x{self.height}")
        print(f"GPU Device: {'CUDA' if GPU_AVAILABLE else 'CPU (fallback)'}")
        print("="*80 + "\n")
        
        # Display all figures at once
        for fig in figures:
            fig.show()
        
        # Block until all windows are closed

        plt.show()


    
    def dual_interactive_explorer(self):
        """Run CPU and GPU explorers side-by-side"""
        print("\n" + "="*70)
        print("  DUAL INTERACTIVE EXPLORER")
        print("="*70)
        print("\nLaunching synchronized CPU and GPU explorers...")
        print("Note: Both explorers will maintain the same coordinates")
        print("Metrics will be collected for each zoom/pan interaction")
        print("="*70 + "\n")
        
        # Clear previous interactive metrics
        self.interactive_metrics = []
        interaction_count = [0]  # Use list to allow modification in nested function
        
        # Create dual explorer
        fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Initial bounds
        xmin, xmax = -2.5, 1.0
        ymin, ymax = -1.25, 1.25
        
        # Renderers
        cpu_renderer = MandelbrotCPU(self.width, self.height, 256)
        gpu_renderer = MandelbrotRenderer(self.width, self.height, 256)
        
        # Initial render
        M_cpu = cpu_renderer.compute(xmin, xmax, ymin, ymax)
        if GPU_AVAILABLE:
            M_gpu = gpu_renderer.mandelbrot_gpu(xmin, xmax, ymin, ymax)
        else:
            M_gpu = gpu_renderer.mandelbrot_cpu(xmin, xmax, ymin, ymax)
        
        im_cpu = ax_cpu.imshow(M_cpu, extent=[xmin, xmax, ymin, ymax], 
                               cmap='hot', origin='lower')
        im_gpu = ax_gpu.imshow(M_gpu, extent=[xmin, xmax, ymin, ymax], 
                               cmap='hot', origin='lower')
        
        ax_cpu.set_title(f'CPU - {cpu_renderer.metrics.time_taken*1000:.1f} ms', 
                        fontweight='bold')
        device_label = "GPU" if GPU_AVAILABLE else "CPU"
        gpu_time = gpu_renderer.gpu_metrics.time_taken if GPU_AVAILABLE else gpu_renderer.cpu_metrics.time_taken
        ax_gpu.set_title(f'{device_label} - {gpu_time*1000:.1f} ms', 
                        fontweight='bold')
        
        fig.suptitle('Dual Interactive Explorer\nLeft-click: Zoom In | Right-click: Zoom Out', 
                    fontsize=14, fontweight='bold')
        
        # Store state
        state = {
            'xmin': xmin, 'xmax': xmax,
            'ymin': ymin, 'ymax': ymax
        }
        
        # Store initial metrics
        cpu_time = cpu_renderer.metrics.time_taken
        gpu_time = gpu_renderer.gpu_metrics.time_taken if GPU_AVAILABLE else gpu_renderer.cpu_metrics.time_taken
        self.interactive_metrics.append({
            'interaction': 0,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time if gpu_time > 0 else 0
        })
        
        def zoom_at(x, y, zoom_factor):
            """Zoom both views"""
            x_range = (state['xmax'] - state['xmin']) * zoom_factor / 2
            y_range = (state['ymax'] - state['ymin']) * zoom_factor / 2
            
            state['xmin'] = x - x_range
            state['xmax'] = x + x_range
            state['ymin'] = y - y_range
            state['ymax'] = y + y_range
            
            # Recompute both
            M_cpu = cpu_renderer.compute(state['xmin'], state['xmax'], 
                                        state['ymin'], state['ymax'])
            if GPU_AVAILABLE:
                M_gpu = gpu_renderer.mandelbrot_gpu(state['xmin'], state['xmax'], 
                                                   state['ymin'], state['ymax'])
            else:
                M_gpu = gpu_renderer.mandelbrot_cpu(state['xmin'], state['xmax'], 
                                                   state['ymin'], state['ymax'])
            
            # Update plots
            im_cpu.set_data(M_cpu)
            im_cpu.set_extent([state['xmin'], state['xmax'], state['ymin'], state['ymax']])
            im_gpu.set_data(M_gpu)
            im_gpu.set_extent([state['xmin'], state['xmax'], state['ymin'], state['ymax']])
            
            # Get timing metrics
            cpu_time = cpu_renderer.metrics.time_taken
            gpu_time = gpu_renderer.gpu_metrics.time_taken if GPU_AVAILABLE else gpu_renderer.cpu_metrics.time_taken
            
            # Store interaction metrics
            interaction_count[0] += 1
            self.interactive_metrics.append({
                'interaction': interaction_count[0],
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time / gpu_time if gpu_time > 0 else 0
            })
            
            # Update titles
            ax_cpu.set_title(f'CPU - {cpu_time*1000:.1f} ms | Interaction #{interaction_count[0]}', 
                           fontweight='bold')
            ax_gpu.set_title(f'{device_label} - {gpu_time*1000:.1f} ms | Interaction #{interaction_count[0]}', 
                           fontweight='bold')
            
            ax_cpu.set_xlim(state['xmin'], state['xmax'])
            ax_cpu.set_ylim(state['ymin'], state['ymax'])
            ax_gpu.set_xlim(state['xmin'], state['xmax'])
            ax_gpu.set_ylim(state['ymin'], state['ymax'])
            
            plt.draw()
        
        def on_click(event):
            """Handle clicks on either subplot"""
            if event.inaxes not in [ax_cpu, ax_gpu]:
                return
            
            x, y = event.xdata, event.ydata
            if event.button == 1:  # Left click
                zoom_at(x, y, 0.5)
            elif event.button == 3:  # Right click
                zoom_at(x, y, 2.0)
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        # After closing, automatically show analysis if there are interactions
        if len(self.interactive_metrics) > 1:  # More than just initial render
            print("\n" + "="*70)
            print("Opening comparative analysis...")
            print("="*70)
            self.comparative_analysis()
    
    def comparative_analysis(self):
        """Generate comparative analysis graphs"""
        print("\n" + "="*70)
        print("  COMPARATIVE ANALYSIS")
        print("="*70)
        
        if not self.interactive_metrics:
            print("\n⚠ No interactive metrics collected yet.")
            print("   Run option 2 (Dual interactive explorer) first and perform some zoom/pan interactions.")
            print("   Then run this analysis to see CPU vs GPU performance for each interaction.")
            return
        
        # Extract data from interactive metrics
        interactions = [m['interaction'] for m in self.interactive_metrics]
        cpu_times = [m['cpu_time'] * 1000 for m in self.interactive_metrics]  # Convert to ms
        gpu_times = [m['gpu_time'] * 1000 for m in self.interactive_metrics]
        speedups = [m['speedup'] for m in self.interactive_metrics]
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        fig.suptitle(f'Interactive Explorer Performance Analysis ({len(interactions)} interactions)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Computation Time per Interaction
        ax1 = axes[0, 0]
        ax1.plot(interactions, cpu_times, marker='o', linewidth=2, markersize=6, 
                label='CPU', color='#FF6B6B', alpha=0.8)
        ax1.plot(interactions, gpu_times, marker='s', linewidth=2, markersize=6, 
                label='GPU', color='#4ECDC4', alpha=0.8)
        ax1.set_xlabel('Interaction Number', fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        ax1.set_title('Render Time per Interaction', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup per Interaction
        ax2 = axes[0, 1]
        colors = ['#4ECDC4' if s >= 1 else '#FF6B6B' for s in speedups]
        bars = ax2.bar(interactions, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='1x (No speedup)', zorder=3)
        
        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_xlabel('Interaction Number', fontweight='bold')
        ax2.set_ylabel('Speedup Factor (CPU time ÷ GPU time)', fontweight='bold')
        ax2.set_title('GPU Speedup per Interaction\n(Higher = GPU Faster)', fontweight='bold', fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        
        # 3. Cumulative Time Comparison
        ax3 = axes[1, 0]
        cumulative_cpu = np.cumsum(cpu_times)
        cumulative_gpu = np.cumsum(gpu_times)
        ax3.plot(interactions, cumulative_cpu, marker='o', linewidth=2, markersize=6,
                label='CPU (cumulative)', color='#FF6B6B', alpha=0.8)
        ax3.plot(interactions, cumulative_gpu, marker='s', linewidth=2, markersize=6,
                label='GPU (cumulative)', color='#4ECDC4', alpha=0.8)
        ax3.set_xlabel('Interaction Number', fontweight='bold')
        ax3.set_ylabel('Cumulative Time (ms)', fontweight='bold')
        ax3.set_title('Cumulative Render Time', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        avg_cpu_time = np.mean(cpu_times)
        avg_gpu_time = np.mean(gpu_times)
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        total_cpu_time = np.sum(cpu_times)
        total_gpu_time = np.sum(gpu_times)
        
        summary_text = (
            f"Total Interactions: {len(interactions)}\n\n"
            f"Average CPU Time: {avg_cpu_time:.2f} ms\n"
            f"Average GPU Time: {avg_gpu_time:.2f} ms\n\n"
            f"Total CPU Time: {total_cpu_time:.2f} ms\n"
            f"Total GPU Time: {total_gpu_time:.2f} ms\n\n"
            f"Average Speedup: {avg_speedup:.2f}x\n"
            f"Max Speedup: {max_speedup:.2f}x\n"
            f"Min Speedup: {min_speedup:.2f}x\n\n"
            f"Time Saved: {total_cpu_time - total_gpu_time:.2f} ms\n\n"
            f"GPU Device: {'CUDA' if GPU_AVAILABLE else 'CPU (fallback)'}"
        )
        
        ax4.text(0.05, 0.6, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1))
        
        plt.tight_layout()
        plt.savefig('interactive_performance_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Analysis saved to interactive_performance_analysis.png")
        plt.show()
        
        # Print summary to console
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)
        print(f"Total Interactions: {len(interactions)}")
        print(f"Average CPU Time: {avg_cpu_time:.2f} ms")
        print(f"Average GPU Time: {avg_gpu_time:.2f} ms")
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Time Saved: {total_cpu_time - total_gpu_time:.2f} ms")
        print("="*70)

    
    def run_all(self):
        """Run all comparison modes"""
        print("\n" + "="*70)
        print("  RUNNING ALL COMPARISON MODES")
        print("="*70)
        
        # 1. Side-by-side comparison
        self.side_by_side_comparison()
        
        # 2. Comparative analysis
        self.comparative_analysis()
        
        # 3. Dual interactive explorer
        print("\nLaunching dual interactive explorer...")
        print("Close the window when done to complete the analysis.")
        self.dual_interactive_explorer()
        
        print("\n[OK] All comparisons complete!")
    
    def create_animation(self):
        """Create zoom animation"""
        print("\n" + "="*70)
        print("  CREATE ZOOM ANIMATION")
        print("="*70)
        
        # Import animation script
        import subprocess
        import sys
        
        print("\nGenerating zoom animation...")
        print("This will create zoom.gif using GPU acceleration.")
        print("="*70 + "\n")
        
        # Run the animation script
        result = subprocess.run([sys.executable, 'make_animation.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n[OK] Animation created successfully!")
            print(f"[OK] Saved as: {os.path.join(self.gifs_dir, 'zoom.gif')}")
        else:
            print("\n[WARNING] Animation creation failed.")
            print(result.stderr)
    
    def create_advanced_animation(self):
        """Create zoom animation with advanced rendering effects"""
        import imageio
        
        print("\n" + "="*70)
        print("  ADVANCED ZOOM ANIMATION")
        print("="*70)
        print("Create stunning animations with advanced rendering effects:")
        print("  • Stripe shading for organic textures")
        print("  • Step shading for contour effects")
        print("  • Blinn-Phong lighting for 3D appearance")
        print("  • Custom color schemes")
        print("="*70)
        
        # Animation settings
        width, height = 600, 450
        max_iter = 5000
        frames = 150
        
        # Target point for zoom
        target_x = -1.749705768080503
        target_y = -6.13369029080495e-05
        
        # Zoom range
        initial_width = 3.0
        final_width = 0.000001
        
        # Get advanced rendering parameters
        print("\nConfigure rendering parameters (press Enter for defaults):\n")
        
        # RGB thetas
        print("RGB Thetas (color scheme):")
        print("  Examples: 0.0 0.15 0.25 (blue-green), 0.85 0.0 0.15 (vibrant)")
        rgb_input = input("  RGB Thetas (3 values, default=0.0 0.15 0.25): ").strip()
        if rgb_input:
            try:
                rgb_values = [float(x) for x in rgb_input.split()]
                rgb_thetas = tuple(rgb_values) if len(rgb_values) == 3 else (0.0, 0.15, 0.25)
            except:
                rgb_thetas = (0.0, 0.15, 0.25)
        else:
            rgb_thetas = (0.0, 0.15, 0.25)
        
        # ncycle
        ncycle_input = input("\nColor cycle frequency (default=32): ").strip()
        ncycle = int(ncycle_input) if ncycle_input else 32
        
        # Stripe shading
        stripe_input = input("Stripe frequency (0-32, 0=none, default=0): ").strip()
        stripe_s = float(stripe_input) if stripe_input else 0
        stripe_sig = 0.9
        
        # Step shading
        step_input = input("Step frequency (0-100, 0=none, default=0): ").strip()
        step_s = float(step_input) if step_input else 0
        
        # Lighting
        print("\nLighting:")
        azimuth_input = input("  Azimuth angle (0-360, default=45): ").strip()
        elevation_input = input("  Elevation angle (0-90, default=45): ").strip()
        
        azimuth = float(azimuth_input) if azimuth_input else 45.
        elevation = float(elevation_input) if elevation_input else 45.
        
        # Convert light angles
        light = np.array([azimuth, elevation, 0.75, 0.2, 0.5, 0.5, 20], dtype=float)
        light[0] = 2 * math.pi * light[0] / 360
        light[1] = math.pi / 2 * light[1] / 90
        
        # Output filename
        output_file = input("\nOutput filename (default=zoom_advanced.gif): ").strip()
        if not output_file:
            output_file = "zoom_advanced.gif"
        
        output_path = os.path.join(self.gifs_dir, output_file)
        
        print("\n" + "="*70)
        print("  RENDERING ANIMATION")
        print("="*70)
        print(f"Frames: {frames}")
        print(f"Resolution: {width}×{height}")
        print(f"Iterations: {max_iter}")
        print(f"RGB Thetas: {rgb_thetas}")
        print(f"ncycle: {ncycle}")
        print(f"Stripe frequency: {stripe_s}")
        print(f"Step frequency: {step_s}")
        print(f"Lighting: Azimuth={azimuth}°, Elevation={elevation}°")
        print("="*70 + "\n")
        
        # Generate color table
        colortable = sin_colortable_advanced(rgb_thetas, ncol=4096)
        ncycle_sqrt = math.sqrt(ncycle)
        
        # Log scale for smooth zoom
        scales = np.logspace(np.log10(initial_width), np.log10(final_width), frames)
        
        images = []
        
        try:
            for i, scale in enumerate(scales):
                print(f"Rendering frame {i+1}/{frames}... ({(i+1)/frames*100:.1f}%)", end='\r')
                
                # Calculate bounds
                xmin = target_x - scale/2
                xmax = target_x + scale/2
                ymin = target_y - (scale * height / width)/2
                ymax = target_y + (scale * height / width)/2
                
                # Compute diagonal for DEM normalization
                diag = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
                
                # Create coordinate arrays
                creal = np.linspace(xmin, xmax, width)
                cim = np.linspace(ymin, ymax, height)
                
                # Render with advanced effects
                img_array = compute_set_advanced(creal, cim, max_iter, colortable, 
                                                ncycle_sqrt, stripe_s, stripe_sig, 
                                                step_s, diag, light)
                
                # Convert to uint8
                img_uint8 = (255 * img_array).astype(np.uint8)
                img_uint8 = img_uint8[::-1, :, :]  # Flip vertically
                images.append(img_uint8)
            
            print("\n\nCompiling GIF...")
            imageio.mimsave(output_path, images, fps=8, loop=0)
            
            print("\n" + "="*70)
            print(f"[OK] Animation saved to: {output_path}")
            print(f"[OK] Total frames: {frames}")
            print(f"[OK] Duration: {frames/8:.1f} seconds")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Animation failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    def render_custom_coordinates(self):
        """Render Mandelbrot at custom coordinates"""
        import json
        import os
        
        presets_file = 'coordinate_presets.json'
        
        print("\n" + "="*70)
        print("  RENDER CUSTOM COORDINATES")
        print("="*70)
        
        # Load presets if available
        presets = {}
        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r') as f:
                    presets = json.load(f)
            except:
                pass
        
        # Show available presets
        if presets:
            print("\nAvailable presets:")
            print("-" * 70)
            for name, data in presets.items():
                print(f"  {name}: {data.get('description', 'No description')}")
            print("-" * 70)
        
        print("\nOptions:")
        print("  1. Use a preset")
        print("  2. Enter custom coordinates")
        print("  3. Save current view as preset (from interactive explorer)")
        print("="*70 + "\n")
        
        choice = input("Enter choice (1-3): ").strip()
        
        try:
            xmin, xmax, ymin, ymax = None, None, None, None
            max_iter = 5000
            rgb_thetas = (0.85, 0.0, 0.15)  # Default RGB thetas
            ncycle = 32
            stripe_s = 0
            step_s = 0
            filename = "custom_render.png"
            use_advanced = 'n'  # Initialize to prevent UnboundLocalError
            
            if choice == '1' and presets:
                # Use preset
                preset_name = input("\nEnter preset name: ").strip()
                if preset_name in presets:
                    preset = presets[preset_name]
                    xmin, xmax, ymin, ymax = preset['coords']
                    max_iter = preset.get('max_iter', 5000)
                    rgb_thetas = tuple(preset.get('rgb_thetas', [0.85, 0.0, 0.15]))
                    ncycle = preset.get('ncycle', 32)
                    stripe_s = preset.get('stripe_s', 0)
                    step_s = preset.get('step_s', 0)
                    filename = os.path.join(self.images_dir, f"{preset_name}.png")
                    print(f"\n[OK] Loaded preset: {preset.get('description', preset_name)}")
                    print(f"[OK] RGB Thetas: {rgb_thetas}")
                    print(f"[OK] Advanced: ncycle={ncycle}, stripe_s={stripe_s}, step_s={step_s}")
                else:
                    print(f"\n[WARNING] Preset '{preset_name}' not found.")
                    return
                    
            elif choice == '2':
                # Custom coordinates
                print("\nEnter coordinates [xmin, xmax, ymin, ymax]:")
                print("Example: -0.5503295086752807 -0.5503293049351449 -0.6259346555912755 -0.625934541001796")
                coords_input = input("\nCoordinates (space-separated): ").strip()
                coords = [float(x) for x in coords_input.split()]
                
                if len(coords) != 4:
                    print("\n[WARNING] Error: Please provide exactly 4 coordinates.")
                    return
                
                xmin, xmax, ymin, ymax = coords
                
                # Ask if user wants to save as preset
                save_preset = input("\nSave as preset? (y/n): ").strip().lower()
                if save_preset == 'y':
                    preset_name = input("Preset name: ").strip()
                    description = input("Description (optional): ").strip()
                    
                    # Ask for RGB thetas
                    print("\nRGB Thetas for color scheme (press Enter for default 0.85 0.0 0.15):")
                    rgb_input = input("RGB Thetas (space-separated, 3 values 0-1): ").strip()
                    if rgb_input:
                        rgb_values = [float(x) for x in rgb_input.split()]
                        if len(rgb_values) == 3:
                            rgb_thetas = tuple(rgb_values)
                        else:
                            print("[WARNING] Invalid RGB thetas, using default")
                    
                    # Ask for advanced parameters
                    print("\nAdvanced rendering parameters (press Enter for defaults):")
                    ncycle_save = input("  ncycle (default=32): ").strip()
                    stripe_save = input("  stripe_s (default=0): ").strip()
                    step_save = input("  step_s (default=0): ").strip()
                    
                    presets[preset_name] = {
                        "coords": [xmin, xmax, ymin, ymax],
                        "max_iter": 5000,
                        "rgb_thetas": list(rgb_thetas),
                        "ncycle": int(ncycle_save) if ncycle_save else 32,
                        "stripe_s": float(stripe_save) if stripe_save else 0,
                        "step_s": float(step_save) if step_save else 0,
                        "description": description if description else f"Custom coordinates"
                    }
                    
                    with open(presets_file, 'w') as f:
                        json.dump(presets, f, indent=2)
                    print(f"[OK] Saved preset '{preset_name}' with RGB thetas {rgb_thetas}")
                    
            elif choice == '3':
                print("\n[WARNING] This feature requires running the interactive explorer first.")
                print("   Use option 2 to explore, then note the coordinates you want to save.")
                return
            else:
                print("\n[WARNING] Invalid choice.")
                return
            
            # Default lighting parameters
            stripe_sig = 0.9  # Stripe smoothness parameter
            light = (45., 45., 0.75, 0.2, 0.5, 0.5, 20)
            
            # For presets (choice == '1'): use preset values directly without prompts
            if choice == '1':
                # Use all preset values as-is, no prompts needed
                print(f"\n[OK] Using preset values:")
                print(f"     Max iterations: {max_iter}")
                print(f"     RGB Thetas: {rgb_thetas}")
                print(f"     ncycle: {ncycle}, stripe_s: {stripe_s}, step_s: {step_s}")
            else:
                # For custom coordinates (choice == '2'): show all prompts
                print("\nOptional parameters (press Enter for defaults):")
                max_iter_input = input(f"Max iterations (default={max_iter}): ").strip()
                if max_iter_input:
                    max_iter = int(max_iter_input)
                
                print(f"\nRGB Thetas for color scheme (current: {rgb_thetas}):")
                print("Examples: 0.0 0.15 0.25 (blue-green), 0.85 0.0 0.15 (vibrant), 0.5 0.7 0.9 (warm)")
                rgb_input = input("RGB Thetas (space-separated, 3 values 0-1, or press Enter): ").strip()
                if rgb_input:
                    rgb_values = [float(x) for x in rgb_input.split()]
                    if len(rgb_values) == 3:
                        rgb_thetas = tuple(rgb_values)
                    else:
                        print("[WARNING] Invalid RGB thetas, using current values")
                
                # Advanced rendering options
                print("\n" + "="*70)
                print("  ADVANCED RENDERING OPTIONS")
                print("="*70)
                print("Enable advanced features for enhanced visual effects:")
                print("  - Stripe shading: Adds organic texture patterns")
                print("  - Step shading: Creates contour-like bands")
                print("  - Blinn-Phong lighting: Realistic 3D lighting effects")
                print("="*70)
                
                use_advanced = input("\nUse advanced rendering? (y/n, default=n): ").strip().lower()
                
                if use_advanced == 'y':
                    print("\nAdvanced parameters (press Enter for defaults):")
                    ncycle_input = input(f"  Color cycle frequency (current={ncycle}): ").strip()
                    if ncycle_input:
                        ncycle = int(ncycle_input)
                    
                    stripe_input = input(f"  Stripe frequency (0-32, current={stripe_s}): ").strip()
                    if stripe_input:
                        stripe_s = float(stripe_input)
                    
                    step_input = input(f"  Step frequency (0-100, current={step_s}): ").strip()
                    if step_input:
                        step_s = float(step_input)
                    
                    print("\n  Lighting (press Enter for defaults):")
                    azimuth_input = input("    Azimuth angle (0-360, default=45): ").strip()
                    elevation_input = input("    Elevation angle (0-90, default=45): ").strip()
                    
                    azimuth = float(azimuth_input) if azimuth_input else 45.
                    elevation = float(elevation_input) if elevation_input else 45.
                    light = (azimuth, elevation, 0.75, 0.2, 0.5, 0.5, 20)
            
            # Ask for output filename (always, for both presets and custom)
            filename_input = input(f"\nOutput filename (default={filename}): ").strip()
            if filename_input:
                filename = filename_input
            
            # Ensure filename is in images directory
            if not filename.startswith(self.images_dir):
                filename = os.path.join(self.images_dir, os.path.basename(filename))
            
            print("\n" + "="*70)
            print("Rendering...")
            print("="*70)
            
            # Render with advanced features if enabled OR for presets (silent rendering)
            # Use Numba JIT for presets to avoid verbose GPU progress output
            use_numba_jit = (use_advanced == 'y') or (choice == '1')
            if use_numba_jit:
                print("[INFO] Using advanced rendering with effects...")
                
                # Generate advanced color table
                colortable = sin_colortable_advanced(rgb_thetas, ncol=4096)
                
                # Convert light angles
                light_converted = np.array(light, dtype=float)
                light_converted[0] = 2 * math.pi * light_converted[0] / 360
                light_converted[1] = math.pi / 2 * light_converted[1] / 90
                
                # Compute diagonal for DEM normalization
                diag = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
                ncycle_sqrt = math.sqrt(ncycle)
                
                # Create coordinate arrays
                creal = np.linspace(xmin, xmax, 1920)
                cim = np.linspace(ymin, ymax, 1080)
                
                start_time = time.time()
                img_array = compute_set_advanced(creal, cim, max_iter, colortable, 
                                                ncycle_sqrt, stripe_s, stripe_sig, 
                                                step_s, diag, light_converted)
                render_time = time.time() - start_time
                
                # Convert to uint8
                img = (255 * img_array).astype(np.uint8)
                img = img[::-1, :, :]  # Flip vertically
                
            else:
                # Standard rendering
                print("[INFO] Using standard rendering...")
                renderer = MandelbrotRenderer(1920, 1080, max_iter)
                if GPU_AVAILABLE:
                    M = renderer.mandelbrot_gpu(xmin, xmax, ymin, ymax)
                    render_time = renderer.gpu_metrics.time_taken
                else:
                    M = renderer.mandelbrot_cpu(xmin, xmax, ymin, ymax)
                    render_time = renderer.cpu_metrics.time_taken
                
                # Apply color scheme with custom RGB thetas
                img = self.apply_color_scheme(M, max_iter, rgb_thetas)
            
            # Save image
            from PIL import Image
            Image.fromarray(img).save(filename)
            
            print(f"\n[OK] Rendered in {render_time*1000:.2f} ms")
            print(f"[OK] Saved as: {filename}")
            print(f"[OK] Resolution: 1920x1080")
            print(f"[OK] Iterations: {max_iter}")
            print(f"[OK] RGB Thetas: {rgb_thetas}")
            if use_advanced == 'y':
                print(f"[OK] Advanced features:")
                print(f"     - ncycle: {ncycle}")
                print(f"     - Stripe frequency: {stripe_s}")
                print(f"     - Step frequency: {step_s}")
                print(f"     - Lighting: Azimuth={light[0]}°, Elevation={light[1]}°")
            print(f"[OK] Coordinates: [{xmin:.10f}, {xmax:.10f}, {ymin:.10f}, {ymax:.10f}]")
            print("="*70)
            
        except ValueError:
            print("\n[WARNING] Error: Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"\n[WARNING] Error: {e}")




def main():
    """Main menu interface"""
    print("\n" + "="*70)
    print("  MANDELBROT: CPU vs GPU COMPARISON TOOL")
    print("="*70)
    print(f"\nGPU Available: {GPU_AVAILABLE}")
    print("="*70)
    
    tool = ComparisonTool(width=800, height=600)
    
    while True:
        print("\n" + "="*70)
        print("  MENU")
        print("="*70)
        print("1. Side-by-side CPU vs GPU comparison (256, 512, 1024, 2048 iterations)")
        print("2. Dual interactive explorer (CPU and GPU synchronized)")
        print("3. Comparative analysis (graphs and metrics)")
        print("4. Create zoom animation (advanced rendering)")
        print("5. Render custom coordinates")
        print("6. Run ALL comparisons (options 1-3)")
        print("7. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            tool.side_by_side_comparison()
        elif choice == '2':
            tool.dual_interactive_explorer()
        elif choice == '3':
            tool.comparative_analysis()
        elif choice == '4':
            tool.create_advanced_animation()
        elif choice == '5':
            tool.render_custom_coordinates()
        elif choice == '6':
            tool.run_all()
        elif choice == '7':
            print("\n[OK] Exiting. Thank you!")
            break
        else:
            print("\n[WARNING] Invalid choice. Please try again.")



if __name__ == "__main__":
    main()
