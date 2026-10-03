"""
Validation of public inputs and logic for dispatching into the supported backends
"""

import math
import sys
from typing import Any

import numpy as npy

from ._cpu import _canny_cpu

# Python 3.10 compatability until Ubuntu 22.04 standard support by Canonical EOL's
if sys.version_info >= (3, 11):
    from enum import StrEnum, auto
else:
    from backports.strenum import StrEnum, auto


_GAUSSIAN_TRUNCATE: float = 4.0
_LOW_THRESH_TYPE_STR: str = "low"
_HIGH_THRESH_TYPE_STR: str = "high"


class CannyBackend(StrEnum):
    """
    Supported backends used for execution for the Canny Operator
    """

    AUTO = auto()
    CPU = auto()
    GPU = auto()


def _gaussian_radius(sigma: float, truncate: float) -> int:
    """
    Formula borrowed from SciPy/CuPyX internals for consistency
    Args:
        sigma: the size of each standard deviation of the gaussian blur kernel used
        truncate: number of standard deviations included before truncating the blur filter
    Returns:
        integer Gaussian support radius
    """
    return int(sigma * truncate + 0.5)


def _validate_threshold_range(threshold: float, threshold_type_str: str) -> None:
    """
    Validate that the threshold passed into this function is a floating point value between
        0.0 and 1.0 inclusive
    Args:
        threshold: the threshold to be validated
        threshold_type_str: threshold name used to identify the invalid argument
    Returns: None
    Raises:
        TypeError: if threshold is a Boolean value
        ValueError: if threshold is non-finite or outside [0.0, 1.0]
    """
    if isinstance(threshold, bool):
        raise TypeError(
            f"ERROR: got a 'bool' value for {threshold_type_str}_threshold "
            f"of {threshold}; expected a floating-point {threshold_type_str}_threshold "
            "between 0.0 and 1.0!"
        )
    if not math.isfinite(threshold):
        raise ValueError(
            f"ERROR: got a non-finite {threshold_type_str}_threshold value "
            f"of {threshold}; expected a floating-point {threshold_type_str}_threshold "
            "between 0.0 and 1.0!"
        )
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"ERROR: got a value of {threshold} for {threshold_type_str}_threshold; "
            f"expected a floating-point {threshold_type_str}_threshold between 0.0 and 1.0!"
        )


def _validate_shared_canny_args(
    sigma: float, low_threshold: float, high_threshold: float
) -> None:
    """
    Validate scalar Canny parameters
    Args:
        sigma: size of each standard devation of the gaussian blur kernel
        low_threshold: the floating point minium absolute pixel value of
            what counts as a weak edge
        high_threshold: the floating point minimum absolute pixel value of
            what counts as a strong edge
    Returns: None
    Raises:
        TypeError: if sigma or either threshold is a Boolean value
        ValueError: if sigma or either threshold is outside its accepted range,
            is non-finite, or if low_threshold is greater than high_threshold
    """
    if isinstance(sigma, bool):
        raise TypeError(
            f"ERROR: got a 'bool' value for sigma of {sigma}; "
            "expected finite nonnegative floating-point sigma!"
        )
    if not math.isfinite(sigma):
        raise ValueError(
            f"ERROR: got a non-finite sigma value of {sigma}; "
            "expected finite nonnegative floating-point sigma!"
        )
    if sigma < 0.0:
        raise ValueError(
            f"ERROR: got a negative sigma value of {sigma}; "
            "expected finite nonnegative floating-point sigma!"
        )
    _validate_threshold_range(low_threshold, _LOW_THRESH_TYPE_STR)
    _validate_threshold_range(high_threshold, _HIGH_THRESH_TYPE_STR)
    if low_threshold > high_threshold:
        raise ValueError(
            f"Expected low_threshold value of {low_threshold} "
            "to be less than or equal to high_threshold "
            f"value of {high_threshold}!"
        )


def _detect_array_backend(image: Any) -> CannyBackend:
    """
    Detect the backend that owns an accepted input array.

    Args:
        image: candidate backend-native image array
    Returns:
        CPU for a NumPy array or GPU for an already-imported CuPy array
    Raises:
        TypeError: if image is not an ndarray from a recognized backend
    """
    if isinstance(image, npy.ndarray):
        return CannyBackend.CPU

    cupy_module = sys.modules.get("cupy")
    cupy_array_type = getattr(cupy_module, "ndarray", None)
    if cupy_array_type is not None and isinstance(image, cupy_array_type):
        return CannyBackend.GPU

    raise TypeError("image must be a NumPy or CuPy ndarray!")


def _resolve_canny_backend(
    image: Any, backend: CannyBackend | str = "auto"
) -> CannyBackend:
    """
    Args:
        image: the input image for which to detect the ndarray backend
        backend: the backend explicitly requested for the image
    Returns:
        The execution backend consistent with both image and backend without
        transferring image data
    Raises:
        ValueError: Unsupported backend
        TypeError: backend mismatch between image and backend OR image itself is not
            using a supported ndarray backend
        RuntimeError: gpu backend not available when it is requested explicitly

    """
    requested_backend: CannyBackend
    try:
        requested_backend = CannyBackend(backend)
    except (TypeError, ValueError) as excpt:
        raise ValueError(f"Unsupported backend: {backend!r}!") from excpt

    selected_backend: CannyBackend
    resident_backend = _detect_array_backend(image)

    if requested_backend == CannyBackend.AUTO:
        selected_backend = resident_backend
    else:
        selected_backend = requested_backend

    if selected_backend != resident_backend:
        raise TypeError(
            f"{selected_backend!r} backend cannot process "
            f"{resident_backend!r}-resident input!"
        )

    if selected_backend is CannyBackend.GPU:
        raise RuntimeError("GPU backend is unavailable!")

    return selected_backend


def canny(  # pylint: disable=too-many-arguments # Arguments implement the reviewed public operator contract
    image: Any,
    sigma: float = 1.5,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    *,
    backend: CannyBackend | str = CannyBackend.AUTO,
    ccl: str | None = None,
    workspace: Any = None,
) -> Any:
    """
    Validate shared Canny arguments and dispatch once to the selected backend.

    Args:
        image: backend-native two-dimensional float32 grayscale image
        sigma: nonnegative Gaussian standard deviation
        low_threshold: absolute weak-edge threshold in the interval [0.0, 1.0]
        high_threshold: absolute strong-edge threshold in the interval [0.0, 1.0]
        backend: auto, CPU, or GPU execution selection
        ccl: optional CPU connected-component-labeling implementation name
        workspace: optional workspace compatible with the selected backend
    Returns:
        backend-native two-dimensional Boolean edge array
    Raises:
        TypeError: if image residency or a backend-specific argument is incompatible
        ValueError: if a scalar, backend name, image, workspace, or CCL choice is invalid
        RuntimeError: if an explicitly selected implementation is unavailable
        NotImplementedError: while the refined CPU numerical stages remain unintegrated
    """
    _validate_shared_canny_args(sigma, low_threshold, high_threshold)
    selected_backend = _resolve_canny_backend(image, backend)
    gaussian_radius = _gaussian_radius(sigma, _GAUSSIAN_TRUNCATE)

    if selected_backend is CannyBackend.CPU:
        return _canny_cpu(
            image,
            sigma,
            gaussian_radius,
            low_threshold,
            high_threshold,
            ccl=ccl,
            workspace=workspace,
        )

    raise RuntimeError(f"Backend {selected_backend.value!r} is unavailable!")
