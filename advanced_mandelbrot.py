"""
Advanced Mandelbrot Renderer
Includes all advanced features: stripe shading, step shading, Blinn-Phong lighting,
smooth iteration, custom color schemes, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from PIL import Image
import json
import os

# Try to import GPU support
try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except:
    GPU_AVAILABLE = False


def sin_colortable(rgb_thetas=(0.85, 0.0, 0.15), ncol=2**12):
    """Create sinusoidal color table
    
    Args:
        rgb_thetas: (float, float, float) - phase for each color channel
        ncol: int - number of colors in the output table
    
    Returns:
        ndarray(dtype=float, ndim=2): color table
    """
    def colormap(x, rgb_thetas):
        y = np.column_stack(((x + rgb_thetas[0]) * 2 * math.pi,
                             (x + rgb_thetas[1]) * 2 * math.pi,
                             (x + rgb_thetas[2]) * 2 * math.pi))
        val = 0.5 + 0.5 * np.sin(y)
        return val
    return colormap(np.linspace(0, 1, ncol), rgb_thetas)


@jit
def blinn_phong(normal, light):
    """Blinn-Phong shading algorithm
    
    Args:
        normal: complex - surface normal
        light: tuple - (azimuth, elevation, intensity, k_ambient, k_diffuse, k_specular, shininess)
    
    Returns:
        float: brightness value
    """
    # Normalize normal
    normal = normal / abs(normal)
    
    # Diffuse light (Lambert shading)
    ldiff = (normal.real * math.cos(light[0]) * math.cos(light[1]) +
             normal.imag * math.sin(light[0]) * math.cos(light[1]) +
             1 * math.sin(light[1]))
    ldiff = ldiff / (1 + 1 * math.sin(light[1]))
    
    # Specular light (Blinn-Phong)
    phi_half = (math.pi/2 + light[1]) / 2
    lspec = (normal.real * math.cos(light[0]) * math.sin(phi_half) +
             normal.imag * math.sin(light[0]) * math.sin(phi_half) +
             1 * math.cos(phi_half))
    lspec = lspec / (1 + 1 * math.cos(phi_half))
    lspec = lspec ** light[6]  # shininess
    
    # Combine: ambient + diffuse + specular
    bright = light[3] + light[4] * ldiff + light[5] * lspec
    bright = bright * light[2] + (1 - light[2]) / 2
    return bright


@jit
def smooth_iter(c, maxiter, stripe_s, stripe_sig):
    """Compute smooth iteration count with additional rendering data
    
    Args:
        c: complex - point in complex plane
        maxiter: int - maximum iterations
        stripe_s: float - stripe frequency
        stripe_sig: float - stripe memory parameter
    
    Returns:
        (float, float, float, complex): smooth_iter, stripe_avg, dem, normal
    """
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
def color_pixel(matxy, niter, stripe_a, step_s, dem, normal, colortable,
                ncycle, light):
    """Color a single pixel with all effects
    
    Args:
        matxy: array to write RGB values
        niter: smooth iteration count
        stripe_a: stripe average value
        step_s: step frequency
        dem: distance estimate
        normal: surface normal
        colortable: color lookup table
        ncycle: color cycling frequency
        light: lighting parameters
    """
    ncol = colortable.shape[0] - 1
    niter = math.sqrt(niter) % ncycle / ncycle
    col_i = round(niter * ncol)
    
    def overlay(x, y, gamma):
        if (2 * y) < 1:
            out = 2 * x * y
        else:
            out = 1 - 2 * (1 - x) * (1 - y)
        return out * gamma + x * (1 - gamma)
    
    # Lighting
    bright = blinn_phong(normal, light)
    
    # Distance estimate
    dem = -math.log(dem) / 12
    dem = 1 / (1 + math.exp(-10 * ((2 * dem - 1) / 2)))
    
    # Shaders
    nshader = 0
    shader = 0
    
    # Stripe shading
    if stripe_a > 0:
        nshader += 1
        shader = shader + stripe_a
    
    # Step shading
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
def compute_set(creal, cim, maxiter, colortable, ncycle, stripe_s, stripe_sig,
                step_s, diag, light):
    """Compute Mandelbrot set (CPU version)"""
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3))
    
    for x in range(xpixels):
        for y in range(ypixels):
            c = complex(creal[x], cim[y])
            niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s, stripe_sig)
            if niter > 0:
                color_pixel(mat[y, x, ], niter, stripe_a, step_s, dem / diag,
                           normal, colortable, ncycle, light)
    return mat


class AdvancedMandelbrot:
    """Advanced Mandelbrot renderer with all features"""
    
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.presets_file = 'coordinate_presets.json'
        self.images_dir = 'images'
        
        # Create images directory if it doesn't exist
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
    
    def load_presets(self):
        """Load coordinate presets from JSON"""
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def render_with_features(self, xmin=-2.5, xmax=1.0, ymin=-1.25, ymax=1.25,
                            max_iter=500, rgb_thetas=(0.0, 0.15, 0.25),
                            stripe_s=0, stripe_sig=0.9, step_s=0,
                            light=(45., 45., 0.75, 0.2, 0.5, 0.5, 20),
                            ncycle=32, filename='advanced_render.png'):
        """Render with all advanced features
        
        Args:
            xmin, xmax, ymin, ymax: coordinate bounds
            max_iter: maximum iterations
            rgb_thetas: color phase offsets (R, G, B)
            stripe_s: stripe frequency (0 = no stripes)
            stripe_sig: stripe smoothness (0-1)
            step_s: step/contour frequency (0 = no steps)
            light: (azimuth, elevation, intensity, k_ambient, k_diffuse, k_specular, shininess)
            ncycle: color cycling frequency
            filename: output filename
        """
        print("\n" + "="*70)
        print("  ADVANCED MANDELBROT RENDERING")
        print("="*70)
        print(f"\nResolution: {self.width}×{self.height}")
        print(f"Iterations: {max_iter}")
        print(f"RGB Thetas: {rgb_thetas}")
        print(f"Stripe frequency: {stripe_s}")
        print(f"Step frequency: {step_s}")
        print(f"Lighting: Azimuth={light[0]}°, Elevation={light[1]}°")
        print("="*70)
        
        # Generate color table
        colortable = sin_colortable(rgb_thetas, ncol=4096)
        
        # Convert light angles
        light_converted = np.array(light, dtype=float)
        light_converted[0] = 2 * math.pi * light_converted[0] / 360
        light_converted[1] = math.pi / 2 * light_converted[1] / 90
        
        # Compute diagonal for DEM normalization
        diag = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
        ncycle_sqrt = math.sqrt(ncycle)
        
        # Create coordinate arrays
        creal = np.linspace(xmin, xmax, self.width)
        cim = np.linspace(ymin, ymax, self.height)
        
        print("\nRendering...")
        import time
        start = time.time()
        
        # Compute the set
        img_array = compute_set(creal, cim, max_iter, colortable, ncycle_sqrt,
                               stripe_s, stripe_sig, step_s, diag, light_converted)
        
        elapsed = time.time() - start
        
        # Convert to uint8
        img_array = (255 * img_array).astype(np.uint8)
        
        # Save image in images folder
        filepath = os.path.join(self.images_dir, filename)
        img = Image.fromarray(img_array[::-1, :, :], 'RGB')
        img.save(filepath)
        
        print(f"\n[OK] Rendered in {elapsed:.2f} seconds")
        print(f"[OK] Saved as: {filepath}")
        print("="*70)
        
        return img_array
    
    def render_preset(self):
        """Render using a preset from JSON"""
        presets = self.load_presets()
        
        if not presets:
            print("\n[WARNING] No presets found in coordinate_presets.json")
            return
        
        print("\n" + "="*70)
        print("  AVAILABLE PRESETS")
        print("="*70)
        for name, data in presets.items():
            desc = data.get('description', 'No description')
            print(f"  {name}: {desc}")
        print("="*70)
        
        preset_name = input("\nEnter preset name: ").strip()
        
        if preset_name not in presets:
            print(f"\n[WARNING] Preset '{preset_name}' not found")
            return
        
        preset = presets[preset_name]
        xmin, xmax, ymin, ymax = preset['coords']
        max_iter = preset.get('max_iter', 500)
        rgb_thetas = tuple(preset.get('rgb_thetas', [0.0, 0.15, 0.25]))
        
        # Load advanced parameters from preset (with defaults)
        preset_ncycle = preset.get('ncycle', 32)
        preset_stripe_s = preset.get('stripe_s', 0)
        preset_step_s = preset.get('step_s', 0)
        
        # Ask for rendering features (showing preset defaults)
        print("\n" + "="*70)
        print("  RENDERING OPTIONS")
        print("="*70)
        print(f"Preset has optimized values: ncycle={preset_ncycle}, stripe_s={preset_stripe_s}, step_s={preset_step_s}")
        print("Press Enter to use preset values, or enter custom values:\n")
        
        ncycle_input = input(f"Color cycle frequency (preset={preset_ncycle}): ").strip()
        ncycle = int(ncycle_input) if ncycle_input else preset_ncycle
        
        stripe_input = input(f"Stripe frequency (preset={preset_stripe_s}): ").strip()
        stripe_s = float(stripe_input) if stripe_input else preset_stripe_s
        
        step_input = input(f"Step frequency (preset={preset_step_s}): ").strip()
        step_s = float(step_input) if step_input else preset_step_s
        
        print("\n  Lighting (press Enter for defaults):")
        azimuth = float(input("  Azimuth angle (0-360, default=45): ") or "45")
        elevation = float(input("  Elevation angle (0-90, default=45): ") or "45")
        
        # Use preset name for filename
        filename = f"{preset_name}_advanced.png"
        
        self.render_with_features(
            xmin, xmax, ymin, ymax,
            max_iter=max_iter,
            rgb_thetas=rgb_thetas,
            stripe_s=stripe_s,
            step_s=step_s,
            ncycle=ncycle,
            light=(azimuth, elevation, 0.75, 0.2, 0.5, 0.5, 20),
            filename=filename
        )
    
    def render_custom(self):
        """Render with custom parameters"""
        print("\n" + "="*70)
        print("  CUSTOM RENDERING")
        print("="*70)
        
        print("\nCoordinates (press Enter for full set view):")
        xmin = float(input("  xmin (default=-2.5): ") or "-2.5")
        xmax = float(input("  xmax (default=1.0): ") or "1.0")
        ymin = float(input("  ymin (default=-1.25): ") or "-1.25")
        ymax = float(input("  ymax (default=1.25): ") or "1.25")
        
        max_iter = int(input("\nMax iterations (default=500): ") or "500")
        
        print("\nRGB Thetas (0-1, press Enter for defaults):")
        r = float(input("  R theta (default=0.0): ") or "0.0")
        g = float(input("  G theta (default=0.15): ") or "0.15")
        b = float(input("  B theta (default=0.25): ") or "0.25")
        
        ncycle = int(input("\nColor cycle frequency (default=32): ") or "32")
        stripe_s = float(input("Stripe frequency (0-32, default=0): ") or "0")
        step_s = float(input("Step frequency (0-100, default=0): ") or "0")
        
        print("\nLighting:")
        azimuth = float(input("  Azimuth (0-360, default=45): ") or "45")
        elevation = float(input("  Elevation (0-90, default=45): ") or "45")
        
        filename = input("\nOutput filename (default=custom_render.png): ") or "custom_render.png"
        
        self.render_with_features(
            xmin, xmax, ymin, ymax,
            max_iter=max_iter,
            rgb_thetas=(r, g, b),
            stripe_s=stripe_s,
            step_s=step_s,
            ncycle=ncycle,
            light=(azimuth, elevation, 0.75, 0.2, 0.5, 0.5, 20),
            filename=filename
        )
    
    def showcase_features(self):
        """Render showcase images demonstrating each feature"""
        print("\n" + "="*70)
        print("  FEATURE SHOWCASE")
        print("="*70)
        print("\nGenerating showcase images for each feature...")
        
        base_coords = (-0.7269, -0.7266, 0.1889, 0.1892)
        
        # 1. Basic smooth coloring
        print("\n1. Basic smooth coloring (no effects)...")
        self.render_with_features(
            *base_coords, max_iter=2000,
            rgb_thetas=(0.0, 0.15, 0.25),
            filename='showcase_1_basic.png'
        )
        
        # 2. Stripe shading
        print("\n2. Stripe shading...")
        self.render_with_features(
            *base_coords, max_iter=2000,
            rgb_thetas=(0.0, 0.15, 0.25),
            stripe_s=8, stripe_sig=0.9,
            filename='showcase_2_stripes.png'
        )
        
        # 3. Step shading
        print("\n3. Step/contour shading...")
        self.render_with_features(
            *base_coords, max_iter=2000,
            rgb_thetas=(0.0, 0.15, 0.25),
            step_s=20,
            filename='showcase_3_steps.png'
        )
        
        # 4. Blinn-Phong lighting
        print("\n4. Blinn-Phong lighting...")
        self.render_with_features(
            *base_coords, max_iter=2000,
            rgb_thetas=(0.0, 0.15, 0.25),
            light=(135., 60., 0.9, 0.1, 0.6, 0.7, 30),
            filename='showcase_4_lighting.png'
        )
        
        # 5. All features combined
        print("\n5. All features combined...")
        self.render_with_features(
            *base_coords, max_iter=2000,
            rgb_thetas=(0.0, 0.15, 0.25),
            stripe_s=6, stripe_sig=0.9,
            step_s=15,
            light=(135., 60., 0.9, 0.1, 0.6, 0.7, 30),
            filename='showcase_5_combined.png'
        )
        
        print("\n" + "="*70)
        print("[OK] Showcase complete! Generated 5 images in images/ folder:")
        print(f"  - {os.path.join(self.images_dir, 'showcase_1_basic.png')}")
        print(f"  - {os.path.join(self.images_dir, 'showcase_2_stripes.png')}")
        print(f"  - {os.path.join(self.images_dir, 'showcase_3_steps.png')}")
        print(f"  - {os.path.join(self.images_dir, 'showcase_4_lighting.png')}")
        print(f"  - {os.path.join(self.images_dir, 'showcase_5_combined.png')}")
        print("="*70)


def main():
    """Main menu"""
    print("\n" + "="*70)
    print("  ADVANCED MANDELBROT RENDERER")
    print("="*70)
    print("\nFeatures:")
    print("  • Smooth iteration coloring")
    print("  • Custom RGB color schemes")
    print("  • Stripe average shading")
    print("  • Step/contour shading")
    print("  • Blinn-Phong lighting")
    print("  • Distance estimation")
    print("="*70)
    
    renderer = AdvancedMandelbrot(width=1920, height=1080)
    
    while True:
        print("\n" + "="*70)
        print("  MENU")
        print("="*70)
        print("1. Render from preset (uses coordinate_presets.json)")
        print("2. Render with custom parameters")
        print("3. Generate feature showcase (5 demo images)")
        print("4. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            renderer.render_preset()
        elif choice == '2':
            renderer.render_custom()
        elif choice == '3':
            renderer.showcase_features()
        elif choice == '4':
            print("\n[OK] Exiting. Thank you!")
            break
        else:
            print("\n[WARNING] Invalid choice")


if __name__ == "__main__":
    main()
