"""
Implementation of vectorized double thresholding for Canny edge
   detector vendored from scikit-image TODO: version here
"""
from _backends_bridge import (
    ModulePair,
    NDArrayFloat,
    NDArrayUint8
)

def double_threshold(backend_modules: ModulePair,
                     greyscale_mags_mat: NDArrayFloat,
                     high_threshold: float = 0.2,
                     low_threshold: float = 0.1) -> NDArrayUint8:
    """
    Given an execution backend (backend_modules), greyscale magnitudes
        matrix (greyscale_mags_mat), high raw value threshold (high_threshold),
        and low raw value threshold (low_threshold), computes the double
        thresholded edges map and returns it.

    Returns: double thresholded edges map as matrix of `uint8`
    """
    xpu = backend_modules.xpu_mod
    xpu_ndi = backend_modules.xpu_ndi_mod
    weak_and_strong_edges_mask = greyscale_mags_mat >= low_threshold
    connect_struct = xpu.ones((3, 3), dtype=xpu.bool_)
    labeled_strong_and_weak_edges, num_edges = \
                      xpu_ndi.label(weak_and_strong_edges_mask, connect_struct)
    if num_edges == 0:
        return 255 * weak_and_strong_edges_mask.astype(xpu.uint8)

    strong_edges_only_bool_mask = greyscale_mags_mat >= high_threshold
    strong_edges_labels = \
        xpu.unique(labeled_strong_and_weak_edges[strong_edges_only_bool_mask])
    edges_kept_bool_mask = xpu.zeros((num_edges + 1,), dtype=xpu.bool_)
    edges_kept_bool_mask[strong_edges_labels] = True
    final_edges_bool_mask = edges_kept_bool_mask[labeled_strong_and_weak_edges]
    return 255 * final_edges_bool_mask.astype(xpu.uint8)
