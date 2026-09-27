"""
CPU specific stages code EXCEPT for the double hysteresis logic
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as npy

from ._cpu_ccl import CCLStrategy, resolve_cpu_ccl

# pylint: disable=invalid-name # Current naming scheme is for clarity while maintaining generic-ness
Float32Array2D: TypeAlias = npy.ndarray[tuple[int, int], npy.dtype[npy.float32]]
BoolArray2D: TypeAlias = npy.ndarray[tuple[int, int], npy.dtype[npy.bool_]]
IntegerArray2D: TypeAlias = npy.ndarray[tuple[int, int], npy.dtype[npy.integer]]


@dataclass(frozen=True)
class CpuWorkspace:  # pylint: disable=too-many-instance-attributes # Attributes are required for efficient memory reuse
    """
    Full-image scratch arrays and workspace metadata for CPU backend of Canny operator
    """

    shape: tuple[int, int]
    ccl_strategy: CCLStrategy
    ccl_callable: Callable
    conv_scratch_n_nms_out: Float32Array2D
    gaussian_out: Float32Array2D
    sobel_horz_out: Float32Array2D
    sobel_vert_out: Float32Array2D
    sobel_mags_out: Float32Array2D
    weak_n_strong_mask: BoolArray2D
    strong_only_mask: BoolArray2D
    labeled_edges_out: IntegerArray2D


def create_cpu_workspace(
    shape: tuple[int, int], *, ccl: CCLStrategy | str = CCLStrategy.SCIPY_CCL
) -> CpuWorkspace:
    """
    Constructs the CPU scratch workspace used for the Canny operator
    Args:
        shape: the size of each full image scratch array in the workspace
        ccl: CCL strategy used by the double hysteresis stage of Canny
    Returns:
        the CPU scratch workspace
    """
    if len(shape) != 2:
        raise ValueError(
            "Expected 2 dimensions for each scratch image array in workspace! "
            f"Got {len(shape)} dimensions instead!"
        )
    if not isinstance(shape[0], int) or shape[0] <= 2:
        raise ValueError(
            "Expected at least 3 pixels of height for each scratch image array in workspace! "
            f"Got {shape[0]} pixels instead!"
        )
    if not isinstance(shape[1], int) or shape[1] <= 2:
        raise ValueError(
            "Expected at least 3 pixels of width for each scratch image array in workspace! "
            f"Got {shape[1]} pixels instead!"
        )
    resolved_strategy: CCLStrategy
    try:
        resolved_strategy = CCLStrategy(ccl)
    except (TypeError, ValueError) as excpt:
        raise ValueError(f"Unsupported CPU CCL strategy: {ccl!r}") from excpt

    labeling_callable, labeled_img_dtype = resolve_cpu_ccl(resolved_strategy)
    conv_scratch_n_nms_out = npy.empty(shape, dtype=npy.float32)
    gaussian_out = npy.empty_like(conv_scratch_n_nms_out)
    sobel_horz_out = npy.empty_like(conv_scratch_n_nms_out)
    sobel_vert_out = npy.empty_like(conv_scratch_n_nms_out)
    sobel_mags_out = npy.empty_like(conv_scratch_n_nms_out)
    weak_n_strong_mask = npy.empty(shape, dtype=npy.bool_)
    strong_only_mask = npy.empty_like(weak_n_strong_mask)
    labeled_edges_out = npy.empty(shape, dtype=labeled_img_dtype)

    return CpuWorkspace(
        shape=shape,
        ccl_strategy=resolved_strategy,
        ccl_callable=labeling_callable,
        conv_scratch_n_nms_out=conv_scratch_n_nms_out,
        gaussian_out=gaussian_out,
        sobel_horz_out=sobel_horz_out,
        sobel_vert_out=sobel_vert_out,
        sobel_mags_out=sobel_mags_out,
        weak_n_strong_mask=weak_n_strong_mask,
        strong_only_mask=strong_only_mask,
        labeled_edges_out=labeled_edges_out,
    )
