import numpy as np
import pytest

from mandelbrot_cpu_only import MandelbrotCPU
import mandelbrot_gpu as gpu_module


WIDTH = 32
HEIGHT = 24
MAX_ITER = 32
BOUNDS = (-2.5, 1.0, -1.25, 1.25)


def test_cpu_render_shape_and_dtype():
    renderer = MandelbrotCPU(
        width=WIDTH,
        height=HEIGHT,
        max_iter=MAX_ITER,
    )

    result = renderer.compute(*BOUNDS)

    assert result.shape == (HEIGHT, WIDTH)
    assert np.issubdtype(result.dtype, np.integer)
    assert np.all(result >= 0)
    assert np.all(result < MAX_ITER)


def test_cpu_render_is_deterministic():
    renderer = MandelbrotCPU(
        width=WIDTH,
        height=HEIGHT,
        max_iter=MAX_ITER,
    )

    first = renderer.compute(*BOUNDS)
    second = renderer.compute(*BOUNDS)

    np.testing.assert_array_equal(first, second)


@pytest.mark.skipif(
    not gpu_module.GPU_AVAILABLE,
    reason="CuPy/CUDA is not available",
)
def test_cpu_and_gpu_render_same_result():
    renderer = gpu_module.MandelbrotRenderer(
        width=WIDTH,
        height=HEIGHT,
        max_iter=MAX_ITER,
    )

    cpu_result = renderer.mandelbrot_cpu(*BOUNDS)
    gpu_result = renderer.mandelbrot_gpu(*BOUNDS)

    np.testing.assert_array_equal(cpu_result, gpu_result)
