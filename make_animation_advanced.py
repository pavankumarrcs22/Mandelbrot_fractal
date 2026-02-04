"""
Advanced Mandelbrot Animation with Enhanced Rendering
Uses stripe shading, step shading, and Blinn-Phong lighting for stunning zoom animations
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mandelbrot_gpu import MandelbrotRenderer, GPU_AVAILABLE
from PIL import Image
import imageio
from numba import jit


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


def make_advanced_animation():
    """Create zoom animation with advanced rendering effects"""
    print("\n" + "="*70)
    print("  ADVANCED MANDELBROT ZOOM ANIMATION")
    print("="*70)
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    # Animation settings (matching mandelbrot.py defaults)
    width = 1280
    max_iter = 500
    frames = 150
    oversampling = 3  # 3x3 anti-aliasing like mandelbrot.py
    
    # Target point for zoom
    target_x = -1.749705768080503
    target_y = -6.13369029080495e-05
    
    # Zoom settings (Gaussian-shaped like mandelbrot.py)
    # Zoom scale varies from 0% (s=1) to 30% (s=0.7)
    # Creates smooth acceleration and deceleration
    
    # Advanced rendering parameters
    print("\n" + "="*70)
    print("  RENDERING SETTINGS")
    print("="*70)
    print("Configure advanced rendering parameters (press Enter for defaults):\n")
    
    # RGB thetas
    print("RGB Thetas (color scheme):")
    print("  Examples: 0.0 0.15 0.25 (blue-green), 0.85 0.0 0.15 (vibrant)")
    rgb_input = input("  RGB Thetas (3 values, default=0.0 0.15 0.25): ").strip()
    if rgb_input:
        rgb_values = [float(x) for x in rgb_input.split()]
        rgb_thetas = tuple(rgb_values) if len(rgb_values) == 3 else (0.0, 0.15, 0.25)
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
    gifs_dir = 'gifs'
    if not os.path.exists(gifs_dir):
        os.makedirs(gifs_dir)
    
    output_file = input("\nOutput filename (default=zoom_advanced.gif): ").strip()
    if not output_file:
        output_file = "zoom_advanced.gif"
    
    output_path = os.path.join(gifs_dir, output_file)
    
    print("\n" + "="*70)
    print("  RENDERING ANIMATION")
    print("="*70)
    print(f"Frames: {frames} (forward) + loop frames")
    print(f"Resolution: {width} pixels wide (height auto-calculated)")
    print(f"Iterations: {max_iter}")
    print(f"Oversampling: {oversampling}×{oversampling} (anti-aliasing)")
    print(f"RGB Thetas: {rgb_thetas}")
    print(f"ncycle: {ncycle}")
    print(f"Stripe frequency: {stripe_s}")
    print(f"Step frequency: {step_s}")
    print(f"Lighting: Azimuth={azimuth}°, Elevation={elevation}°")
    print(f"Zoom: Gaussian-shaped (matching mandelbrot.py)")
    print("="*70 + "\n")
    
    # Generate color table
    colortable = sin_colortable_advanced(rgb_thetas, ncol=4096)
    ncycle_sqrt = math.sqrt(ncycle)
    
    # Gaussian-shaped zoom (matching mandelbrot.py)
    # Creates smooth acceleration and deceleration
    def gaussian(n, sig=1):
        x = np.linspace(-1, 1, n)
        return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
    
    zoom_scales = 1 - gaussian(frames, 1/2) * 0.3  # From 0% (s=1) to 30% (s=0.7)
    
    # Initial coordinates (full Mandelbrot set view)
    initial_coord = (-2.6, 1.845, -1.25, 1.25)
    current_coord = list(initial_coord)
    
    images = []
    
    try:
        for i in range(frames):
            print(f"Rendering frame {i+1}/{frames}... ({(i+1)/frames*100:.1f}%)", end='\r')
            
            # Soft zoom at target (matching mandelbrot.py szoom_at method)
            s = zoom_scales[i]
            xrange = (current_coord[1] - current_coord[0]) / 2
            yrange = (current_coord[3] - current_coord[2]) / 2
            
            # Partial centering towards target
            x_center = target_x * (1 - s**2) + (current_coord[1] + current_coord[0])/2 * s**2
            y_center = target_y * (1 - s**2) + (current_coord[3] + current_coord[2])/2 * s**2
            
            # Update coordinates with zoom
            xmin = x_center - xrange * s
            xmax = x_center + xrange * s
            ymin = y_center - yrange * s
            ymax = y_center + yrange * s
            
            current_coord = [xmin, xmax, ymin, ymax]
            
            # Calculate height maintaining aspect ratio
            height = round(width / (xmax - xmin) * (ymax - ymin))
            
            # Compute diagonal for DEM normalization
            diag = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            # Apply oversampling (3x3 like mandelbrot.py)
            width_os = width * oversampling
            height_os = height * oversampling
            
            # Create coordinate arrays
            creal = np.linspace(xmin, xmax, width_os)
            cim = np.linspace(ymin, ymax, height_os)
            
            # Render with advanced effects
            img_array = compute_set_advanced(creal, cim, max_iter, colortable, 
                                            ncycle_sqrt, stripe_s, stripe_sig, 
                                            step_s, diag, light)
            
            # Apply oversampling (average 3x3 blocks)
            if oversampling > 1:
                img_array = (img_array
                           .reshape((height, oversampling, width, oversampling, 3))
                           .mean(3).mean(1))
            
            # Convert to uint8
            img_uint8 = (255 * img_array).astype(np.uint8)
            img_uint8 = img_uint8[::-1, :, :]  # Flip vertically
            images.append(img_uint8)
        
        # Add reverse frames for loop (like mandelbrot.py)
        print("\n\nAdding loop frames...")
        images_with_loop = images + images[::-2]  # Reverse at 2x speed
        
        print("Compiling GIF...")
        imageio.mimsave(output_path, images_with_loop, fps=10, loop=0)
        
        print("\n" + "="*70)
        print(f"[OK] Animation saved to: {output_path}")
        print(f"[OK] Forward frames: {frames}")
        print(f"[OK] Total frames (with loop): {len(images_with_loop)}")
        print(f"[OK] Duration: {len(images_with_loop)/10:.1f} seconds")
        print(f"[OK] Resolution: {width}×{height} (with {oversampling}×{oversampling} oversampling)")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERROR] Animation failed: {e}")
        import traceback


if __name__ == "__main__":
    make_advanced_animation()
