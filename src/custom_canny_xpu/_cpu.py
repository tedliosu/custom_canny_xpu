"""
CPU specific stages code EXCEPT for the double hysteresis logic
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

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


def _validate_cpu_canny_image(image: Any, gaussian_radius: int) -> None:
    """
    Validate CPU residency, dtype, value range, shape, and minimum dimensions.

    Args:
        image: candidate NumPy image
        gaussian_radius: radius of the active Gaussian kernel
    Returns:
        None.
    Raises:
        TypeError: if image is not a NumPy ndarray with float32 dtype
        ValueError: if image has an invalid shape, dimensions, or pixel values
    """
    active_max_radius = max(1, gaussian_radius)
    required_any_dim_size = 2 * active_max_radius
    expected_msg_error_str = (
        "Expected a 2D NumPy float32 ndarray for image where all "
        + "values in said array are between 0.0 and 1.0 inclusive, "
        + f"and width and height are at least each {required_any_dim_size} "
        + "pixels!"
    )

    if not isinstance(image, npy.ndarray):
        raise TypeError(f"ERROR: got an object of type {type(image)} for image! ")
    if image.ndim != 2:
        raise ValueError(
            f"ERROR: got an image with {image.ndim} dimensions! "
            + expected_msg_error_str
        )
    if image.dtype != npy.dtype(npy.float32):
        raise TypeError(
            f"ERROR: got an image of data type {image.dtype.name}! "
            + expected_msg_error_str
        )
    if not npy.isfinite(image).all():
        raise ValueError(
            "ERROR: got a 2D float32 image where not all of the values "
            "are finite!" + expected_msg_error_str
        )
    if (
        npy.less(image, npy.float32(0.0)).any()
        or npy.greater(image, npy.float32(1.0)).any()
    ):
        raise ValueError(
            "ERROR: got a 2D float32 image where not all of the values "
            "are between 0.0 and 1.0 inclusive! " + expected_msg_error_str
        )
    if (
        image.shape[0] < required_any_dim_size
        or image.shape[1] < required_any_dim_size
    ):
        raise ValueError(
            "CPU Canny requires height and width of at least "
            f"{required_any_dim_size} pixels (minimum shape "
            f"({required_any_dim_size}, {required_any_dim_size})) because "
            "custom convolution border handling uses an active maximum "
            f"radius of {active_max_radius}; received shape {image.shape}!"
        )


def create_cpu_workspace(
    shape: tuple[int, int], *, ccl: CCLStrategy | str = CCLStrategy.SCIPY_CCL
) -> CpuWorkspace:
    """
    Constructs the CPU scratch workspace used for the Canny operator

    Args:
        shape: positive two-dimensional size of each full-image scratch array
        ccl: CCL strategy used by the double hysteresis stage of Canny
    Returns:
        the CPU scratch workspace
    Raises:
        ValueError: if the shape or CCL strategy is invalid
        RuntimeError: if an explicitly requested optional CCL is unavailable
    """
    if len(shape) != 2:
        raise ValueError(
            "Expected 2 dimensions for each scratch image array in workspace! "
            f"Got {len(shape)} dimensions instead!"
        )
    # bool subclasses int, but True and False are not meaningful dimensions.
    if (
        not isinstance(shape[0], int)
        or isinstance(shape[0], bool)
        or shape[0] <= 0
    ):
        raise ValueError(
            "Expected a positive integer height for each scratch image array in workspace! "
            f"Got {shape[0]} pixels instead!"
        )
    if (
        not isinstance(shape[1], int)
        or isinstance(shape[1], bool)
        or shape[1] <= 0
    ):
        raise ValueError(
            "Expected a positive integer width for each scratch image array in workspace! "
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
    labeled_edges_out = cast(
        IntegerArray2D, npy.empty(shape, dtype=labeled_img_dtype)
    )

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


def _validate_cpu_workspace_compatibility(  # pylint: disable=too-many-branches # Each branch validates one cheap structural invariant
    workspace: Any, shape: tuple[int, int]
) -> None:
    """
    Validate inexpensive structural requirements for a borrowed CPU workspace.

    Args:
        workspace: candidate CPU workspace supplied by the caller
        shape: required image height and width
    Returns:
        None.
    Raises:
        TypeError: if the workspace or one of its buffers has an invalid type or dtype
        ValueError: if workspace metadata or a buffer shape is incompatible
    """
    if not isinstance(workspace, CpuWorkspace):
        raise TypeError(
            "Expected workspace to be a CpuWorkspace for CPU execution; "
            f"got {type(workspace)}!"
        )
    if workspace.shape != shape:
        raise ValueError(
            f"Expected CPU workspace shape {shape}; got {workspace.shape}!"
        )
    if not isinstance(workspace.ccl_strategy, CCLStrategy):
        raise TypeError("CPU workspace has invalid CCL strategy metadata!")
    if not callable(workspace.ccl_callable):
        raise TypeError("CPU workspace CCL callable metadata is not callable!")

    float_buffers = (
        workspace.conv_scratch_n_nms_out,
        workspace.gaussian_out,
        workspace.sobel_horz_out,
        workspace.sobel_vert_out,
        workspace.sobel_mags_out,
    )
    boolean_buffers = (
        workspace.weak_n_strong_mask,
        workspace.strong_only_mask,
    )

    for buffer in float_buffers:
        if not isinstance(buffer, npy.ndarray) or buffer.dtype != npy.float32:
            raise TypeError("CPU workspace float scratch buffers must be float32 ndarrays!")
        if buffer.shape != shape:
            raise ValueError(
                f"Expected CPU workspace float buffer shape {shape}; got {buffer.shape}!"
            )
        if not buffer.flags.writeable:
            raise ValueError("CPU workspace float scratch buffers must be writable!")

    for buffer in boolean_buffers:
        if not isinstance(buffer, npy.ndarray) or buffer.dtype != npy.bool_:
            raise TypeError("CPU workspace Boolean scratch buffers must be bool ndarrays!")
        if buffer.shape != shape:
            raise ValueError(
                f"Expected CPU workspace Boolean buffer shape {shape}; got {buffer.shape}!"
            )
        if not buffer.flags.writeable:
            raise ValueError("CPU workspace Boolean scratch buffers must be writable!")

    labeled_edges_out = workspace.labeled_edges_out
    if not isinstance(labeled_edges_out, npy.ndarray) or not npy.issubdtype(
        labeled_edges_out.dtype, npy.integer
    ):
        raise TypeError("CPU workspace label buffer must be an integer ndarray!")
    if labeled_edges_out.shape != shape:
        raise ValueError(
            f"Expected CPU workspace label buffer shape {shape}; "
            f"got {labeled_edges_out.shape}!"
        )
    if not labeled_edges_out.flags.writeable:
        raise ValueError("CPU workspace label buffer must be writable!")


def _prepare_cpu_workspace(
    shape: tuple[int, int], *, ccl: str | None, workspace: Any
) -> CpuWorkspace:
    """
    Construct or validate the CPU workspace selected for one Canny call.

    Args:
        shape: validated CPU image height and width
        ccl: optional requested CPU CCL implementation name
        workspace: optional caller-supplied CPU workspace
    Returns:
        compatible CPU workspace owned by the call or borrowed from the caller
    Raises:
        TypeError: if a supplied workspace has an incompatible type or buffer dtype
        ValueError: if CCL and workspace are both supplied or workspace shape is invalid
        RuntimeError: if an explicitly requested optional CCL is unavailable
    """
    if workspace is not None and ccl is not None:
        raise ValueError("ccl and workspace cannot both be supplied explicitly!")

    if workspace is None:
        if ccl is None:
            return create_cpu_workspace(shape)
        return create_cpu_workspace(shape, ccl=ccl)

    _validate_cpu_workspace_compatibility(workspace, shape)
    return workspace


def _execute_cpu_canny(  # pylint: disable=too-many-arguments # Mirrors the validated numerical operator inputs
    image: Float32Array2D,
    sigma: float,
    gaussian_radius: int,
    low_threshold: float,
    high_threshold: float,
    *,
    workspace: CpuWorkspace,
) -> BoolArray2D:
    """
    Execute the refined CPU numerical stages once they are integrated.

    Args:
        image: validated two-dimensional NumPy float32 grayscale image
        sigma: validated Gaussian standard deviation
        gaussian_radius: Gaussian support radius derived from sigma
        low_threshold: validated absolute weak-edge threshold
        high_threshold: validated absolute strong-edge threshold
        workspace: compatible CPU scratch workspace
    Returns:
        two-dimensional NumPy Boolean edge array
    Raises:
        NotImplementedError: until the refined CPU numerical stages are integrated
    """
    _ = (
        image,
        sigma,
        gaussian_radius,
        low_threshold,
        high_threshold,
        workspace,
    )
    raise NotImplementedError("Refined CPU Canny numerical stages are not integrated yet!")


def _canny_cpu(  # pylint: disable=too-many-arguments # Receives the public parameters after one-time dispatch
    image: Any,
    sigma: float,
    gaussian_radius: int,
    low_threshold: float,
    high_threshold: float,
    *,
    ccl: str | None,
    workspace: Any,
) -> BoolArray2D:
    """
    Validate CPU-specific inputs, prepare scratch storage, and execute Canny.

    Args:
        image: candidate CPU-resident two-dimensional float32 grayscale image
        sigma: validated Gaussian standard deviation
        gaussian_radius: Gaussian support radius derived from sigma
        low_threshold: validated absolute weak-edge threshold
        high_threshold: validated absolute strong-edge threshold
        ccl: optional requested CPU CCL implementation name
        workspace: optional caller-supplied CPU workspace
    Returns:
        two-dimensional NumPy Boolean edge array
    Raises:
        TypeError: if the image or workspace is incompatible with CPU execution
        ValueError: if the image, CCL, or workspace is invalid
        RuntimeError: if an explicitly requested optional CCL is unavailable
        NotImplementedError: until the refined CPU numerical stages are integrated
    """
    _validate_cpu_canny_image(image, gaussian_radius)
    cpu_workspace = _prepare_cpu_workspace(
        image.shape,
        ccl=ccl,
        workspace=workspace,
    )
    return _execute_cpu_canny(
        image,
        sigma,
        gaussian_radius,
        low_threshold,
        high_threshold,
        workspace=cpu_workspace,
    )
