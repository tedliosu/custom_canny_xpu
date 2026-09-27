"""
CPU double hysteresis logic dependencies
"""

import sys
from collections.abc import Callable
from typing import Any

import numpy as npy

# Python 3.10 compatability until Ubuntu 22.04 standard support by Canonical EOL's
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class CCLStrategy(StrEnum):
    """
    Supported CCL strategies used by double hysteresis for Canny Operator
    """

    SCIPY_CCL = "scipy"
    CC3D_CCL = "cc3d"


def _resolve_scipy_ccl() -> tuple[Callable, type[npy.signedinteger]]:
    """
    Resolves the scipy CCL labeling function used by Canny and the associated
        dtype used for the array storing the labels used by double hysteresis
        stage
    Returns: Tuple consisting CCL callable and dtype used for scipy based CCL result array
    """

    from scipy.ndimage import label  # pylint: disable=import-outside-toplevel # noqa: I001 # lazily resolving import is intentional

    return label, npy.int32


def _resolve_cc3d_ccl() -> tuple[Callable, type[npy.unsignedinteger]]:

    raise RuntimeError("TODO: Implement optional CC3D custom binding resolution!")


_CCL_RESOLVERS: dict[CCLStrategy, Callable[[], tuple[Callable, Any]]] = {
    CCLStrategy.SCIPY_CCL: _resolve_scipy_ccl,
    CCLStrategy.CC3D_CCL: _resolve_cc3d_ccl,
}


def resolve_cpu_ccl(ccl_strategy: CCLStrategy) -> tuple[Callable, Any]:
    """
    Args:
        ccl_strategy: CCL strategy to be used for double hysteresis stage of Canny
    Returns:
        tuple of labeling callable used for CCL corresponding to the requested CCL strategy
            and the dtype of the array used to store labels in the labeled image
    """
    return _CCL_RESOLVERS[ccl_strategy]()
