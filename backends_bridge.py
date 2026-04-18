"""
Definitions of variables and classes used to bridge custom
    canny pipeline to various execution backends.
"""
from dataclasses import dataclass
from types import ModuleType
import numpy as npy
import scipy.ndimage as npy_ndi

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

BACKENDS_MAP: dict[str, ModulePair] = {}


BACKENDS_MAP["numpy"] = ModulePair(npy, npy_ndi)


try:
    import cupy as cpy
    import cupyx.scipy.ndimage as cpy_ndi
    BACKENDS_MAP["cupy"] = ModulePair(cpy, cpy_ndi)
except ModuleNotFoundError:
    pass
