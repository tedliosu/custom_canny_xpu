"""Backend definitions used by the custom Canny operator."""
from dataclasses import dataclass
from types import ModuleType
from typing import Union, TypeAlias
import sys
from numpy import typing as npy_typing
import numpy as npy
import scipy.ndimage as npy_ndi
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

@dataclass(frozen=True)
class ModulePair:
    """
    Members of this class are pairs of modules,
        where the first module is the baseline
        numpy/cupy/etc. module, and the second
        module is corresponding scipy ndimage
        analog.
    """
    xpu_mod: ModuleType
    xpu_ndi_mod: ModuleType

class SupportedBackends(StrEnum):
    """
    All execution backends supported by this code base
    """
    NPY_NAME: str = "numpy"
    CPY_NAME: str = "cupy"

REGIS_BACKENDS_DICT: dict[str, ModulePair] = {}
REGIS_BACKENDS_DICT[SupportedBackends.NPY_NAME] = \
                              ModulePair(npy, npy_ndi)
NDArrayFloat: TypeAlias = npy_typing.NDArray[npy.floating]
NDArrayUint8: TypeAlias = npy_typing.NDArray[npy.uint8]

try:
    import cupy as cpy
    import cupyx.scipy.ndimage as cpy_ndi
    from cupy import typing as cpy_typing
    REGIS_BACKENDS_DICT[SupportedBackends.CPY_NAME] = \
                                  ModulePair(cpy, cpy_ndi)
    NDArrayFloat = Union[npy_typing.NDArray[npy.floating],
                         cpy_typing.NDArray[cpy.floating]]
    NDArrayUint8 = Union[npy_typing.NDArray[npy.uint8],
                             cpy_typing.NDArray[cpy.uint8]]
except ModuleNotFoundError:
    pass
