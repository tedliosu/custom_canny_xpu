"""
Validation of public inputs and logic for dispatching into the supported backends
"""

import math

import numpy as npy

from ._cpu import Float32Array2D

_GAUSSIAN_TRUNCATE: float = 4.0
_LOW_THRESH_TYPE_STR: str = "low"
_HIGH_THRESH_TYPE_STR: str = "high"


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


def _validate_cpu_canny_image(image: Float32Array2D, gaussian_radius: int) -> None:
    """
    Validates CPU residency, dtype, values range, shape, and CPU dimensions of image
    Args:
        image: object to be validated
        gaussian_radius: radius of gaussian blur kernel to be applied
    Returns: None
    Raises:
        TypeError: if image is not a NumPy ndarray with float32 dtype
        ValueError: if image has an invalid shape, dimensions, or pixel values
    """
    required_any_dim_size: int = 2 * max(1, gaussian_radius)
    expected_msg_error_str: str = (
        "Expected a 2D NumPy float32 ndarray for image where all "
        + "values in said array are between 0.0 and 1.0 inclusive, "
        + f"and width and height are at least each {required_any_dim_size} "
        + "pixels!"
    )

    if not isinstance(image, npy.ndarray):
        raise TypeError(f"ERROR: got an object of type {type(image)} for image! ")
    if not image.ndim == 2:
        raise ValueError(
            f"ERROR: got an image with {image.ndim} dimensions! "
            + expected_msg_error_str
        )
    if not npy.isdtype(image.dtype, npy.float32):
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

    if image.shape[0] < required_any_dim_size:
        raise ValueError(
            "ERROR: got a 2D float32 image where all values are between "
            f"0.0 and 1.0 inclusive, but image height is {image.shape[0]}! "
            + expected_msg_error_str
        )
    if image.shape[1] < required_any_dim_size:
        raise ValueError(
            "ERROR: got a 2D float32 image where all values are between "
            f"0.0 and 1.0 inclusive, but image width is {image.shape[1]}! "
            + expected_msg_error_str
        )
