"""
Unit tests for checking if inputs are validated correctly by the API exposed by
    'custom_canny_xpu' module
"""
# Private helpers are tested directly until public canny validation is available.
# pylint: disable=protected-access

import numpy as npy
import pytest

from custom_canny_xpu import _api
from custom_canny_xpu._cpu import Float32Array2D


@pytest.mark.parametrize(
    "sigma, truncate, expected_radius",
    [
        (0.0, 4.0, 0),
        (0.5, 4.0, 2),
        (1.5, 4.0, 6),
        (2.0, 4.0, 8),
        (0.0, 3.0, 0),
        (1.5, 3.0, 5),
    ],
)
def test_gaussian_radius_calculation_policy(
    sigma: float, truncate: float, expected_radius: int
) -> None:
    """
    Verify that the gaussian radius calculation policy is correct
    Args:
        sigma: size of each standard deviation of the gaussian kernel corresponding to
           the calculated radius
        truncate: number of standard deviations at which to truncate the gaussian kernel
            radius
        expected_radius: expected integer Gaussian support radius
    Returns:
        None.
    """
    assert _api._gaussian_radius(sigma, truncate) == expected_radius


@pytest.mark.parametrize(
    "sigma, low_threshold, high_threshold",
    [(0.0, 0.0, 1.0), (1.5, 0.1, 0.2), (2.0, 0.2, 0.2)],
)
def test_valid_shared_canny_args_acceptance(
    sigma: float, low_threshold: float, high_threshold: float
) -> None:
    """
    Verify that the shared canny arguments validation is correct
    Args:
        sigma: size of each standard deviation of the gaussian kernel used for blurring for Canny
        low_threshold: floating point min abs pix val of what counts as a weak edge
        high_threshold: floating point min abs pix val of what counts as a strong edge
    Returns:
        None.
    """
    _api._validate_shared_canny_args(sigma, low_threshold, high_threshold)


@pytest.mark.parametrize(
    "sigma, low_threshold, high_threshold, expected_exception, message",
    [
        (True, 0.1, 0.2, TypeError, "bool.*sigma"),
        (-0.1, 0.1, 0.2, ValueError, "negative sigma"),
        (npy.inf, 0.1, 0.2, ValueError, "non-finite sigma"),
        (1.5, True, 0.2, TypeError, "bool.*low_threshold"),
        (1.5, -0.1, 0.2, ValueError, "low_threshold"),
        (1.5, 0.1, npy.nan, ValueError, "non-finite high_threshold"),
        (1.5, 0.1, 1.1, ValueError, "high_threshold"),
        (1.5, 0.3, 0.2, ValueError, "less than or equal"),
    ],
)
def test_invalid_shared_canny_args_rejection(
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """
    Verify that invalid shared scalar arguments are rejected.

    Args:
        sigma: invalid or accompanying Gaussian standard deviation
        low_threshold: invalid or accompanying weak-edge threshold
        high_threshold: invalid or accompanying strong-edge threshold
        expected_exception: exception class required by the validation contract
        message: regular expression expected in the exception message
    Returns:
        None.
    """
    with pytest.raises(expected_exception, match=message):
        _api._validate_shared_canny_args(sigma, low_threshold, high_threshold)


@pytest.mark.parametrize(
    "shape, gaussian_radius",
    [
        ((2, 2), 0),
        ((12, 12), 6),
        ((12, 17), 6),
        ((24, 32), 6),
    ],
)
def test_valid_cpu_canny_images_acceptance(
    shape: tuple[int, int], gaussian_radius: int
) -> None:
    """
    Verify that valid boundary and non-square CPU images are accepted.

    Args:
        shape: height and width of the generated valid image
        gaussian_radius: radius determining the minimum supported dimensions
    Returns:
        None.
    """
    image = npy.zeros(shape, dtype=npy.float32)

    _api._validate_cpu_canny_image(image, gaussian_radius)


@pytest.mark.parametrize(
    "image, gaussian_radius, expected_exception, message",
    [
        ([[0.0, 0.0], [0.0, 0.0]], 0, TypeError, "object of type"),
        (npy.zeros((2, 2), dtype=npy.float64), 0, TypeError, "float64"),
        (npy.zeros((2, 2, 1), dtype=npy.float32), 0, ValueError, "3 dimensions"),
        (npy.zeros((0, 12), dtype=npy.float32), 6, ValueError, "height is 0"),
        (npy.zeros((12, 0), dtype=npy.float32), 6, ValueError, "width is 0"),
        (npy.zeros((1, 2), dtype=npy.float32), 0, ValueError, "height is 1"),
        (npy.zeros((2, 1), dtype=npy.float32), 0, ValueError, "width is 1"),
        (npy.zeros((11, 12), dtype=npy.float32), 6, ValueError, "height is 11"),
        (npy.zeros((12, 11), dtype=npy.float32), 6, ValueError, "width is 11"),
    ],
)
def test_invalid_cpu_canny_image_structure_rejection(
    image: object,
    gaussian_radius: int,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """
    Verify that invalid CPU image types, structures, and dimensions are rejected.

    Args:
        image: invalid CPU image candidate supplied by the parametrized case
        gaussian_radius: radius determining the minimum supported dimensions
        expected_exception: exception class required by the validation contract
        message: regular expression expected in the exception message
    Returns:
        None.
    """
    with pytest.raises(expected_exception, match=message):
        _api._validate_cpu_canny_image(image, gaussian_radius)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "image, sigma",
    [
        (npy.array([[0.0, 2.0], [1.0, 0.5]], dtype=npy.float32), 0.0),
        (npy.array([[0.0, -2.0], [1.0, 0.5]], dtype=npy.float32), 0.0),
        (
            npy.array(
                [
                    [0.0, 0.5, npy.inf, 0.2],
                    [1.0, 0.5, 0.3, 0.4],
                    [0.55, 0.12, 0.2, 0.3],
                    [0.3, 0.3, 0.6, 0.7],
                ],
                dtype=npy.float32,
            ),
            0.4,
        ),
        (
            npy.array(
                [
                    [0.0, 0.5, 0.1, 0.2],
                    [1.0, 0.5, 0.3, 0.4],
                    [0.55, npy.nan, 0.2, 0.3],
                    [0.3, 0.3, 0.6, 0.7],
                    [0.3, 0.3, 0.8, 0.7],
                ],
                dtype=npy.float32,
            ),
            0.5,
        ),
    ],
)
def test_validation_rejects_image_pixel_values_out_of_range(
    image: Float32Array2D, sigma: float
) -> None:
    """
    Verify that images containing float32 pixel values that aren't finite
        nor aren't between 0.0 and 1.0 inclusive are rejected

    Args:
        image: invalid image supplied by the paramtrized test case
        sigma: valid sigma accompanying each image supplied by the parametrized test case
    Returns: None
    """
    radius = _api._gaussian_radius(sigma, _api._GAUSSIAN_TRUNCATE)
    with pytest.raises(ValueError):
        _api._validate_cpu_canny_image(image, radius)
