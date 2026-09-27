"""
CPU workspace construction tests.
"""

import numpy as npy
import pytest

from custom_canny_xpu import CpuWorkspace, create_cpu_workspace
from custom_canny_xpu._cpu_ccl import CCLStrategy


def test_create_scipy_workspace() -> None:
    """
    Verify construction of a SciPy-backed CPU workspace.

    Returns:
        None.
    """
    shape = (5, 7)

    workspace = create_cpu_workspace(shape, ccl="scipy")

    assert isinstance(workspace, CpuWorkspace)
    assert workspace.shape == shape
    assert workspace.ccl_strategy is CCLStrategy.SCIPY_CCL

    float_buffers = (
        workspace.conv_scratch_n_nms_out,
        workspace.gaussian_out,
        workspace.sobel_horz_out,
        workspace.sobel_vert_out,
        workspace.sobel_mags_out,
    )
    for buffer in float_buffers:
        assert buffer.shape == shape
        assert buffer.dtype == npy.float32

    boolean_buffers = (
        workspace.weak_n_strong_mask,
        workspace.strong_only_mask,
    )
    for buffer in boolean_buffers:
        assert buffer.shape == shape
        assert buffer.dtype == npy.bool_

    assert workspace.labeled_edges_out.shape == shape
    assert workspace.labeled_edges_out.dtype == npy.int32
    assert callable(workspace.ccl_callable)


def test_no_memory_aliasing_workspace() -> None:
    """
    Verify that creating two copies of the SciPy-backed CPU workspace does not
        create inter-workspaces memory aliasing.
    Returns:
        None
    """
    shape = (9, 7)

    workspace1 = create_cpu_workspace(shape, ccl="scipy")
    workspace2 = create_cpu_workspace(shape, ccl="scipy")

    ws1_buffers = (
        workspace1.conv_scratch_n_nms_out,
        workspace1.gaussian_out,
        workspace1.sobel_horz_out,
        workspace1.sobel_vert_out,
        workspace1.sobel_mags_out,
        workspace1.weak_n_strong_mask,
        workspace1.strong_only_mask,
        workspace1.labeled_edges_out,
    )
    ws2_buffers = (
        workspace2.conv_scratch_n_nms_out,
        workspace2.gaussian_out,
        workspace2.sobel_horz_out,
        workspace2.sobel_vert_out,
        workspace2.sobel_mags_out,
        workspace2.weak_n_strong_mask,
        workspace2.strong_only_mask,
        workspace2.labeled_edges_out,
    )

    for buffer_idx, ref_buffer in enumerate(ws1_buffers):
        assert not npy.shares_memory(ref_buffer, ws2_buffers[buffer_idx])


@pytest.mark.parametrize(
    "shape",
    (
        (2, 3),
        (3, 2),
        (3,),
        (3, 4, 5),
        (3.5, 4),
        (True, 4),
    ),
)
def test_create_cpu_workspace_rejects_invalid_shapes(shape: tuple) -> None:
    """
    Verify that malformed or unsupported workspace shapes are rejected.

    Args:
        shape: invalid shape supplied by the parametrized test case
    Returns:
        None
    """
    with pytest.raises(ValueError):
        create_cpu_workspace(shape)  # type: ignore[arg-type]


def test_create_cpu_workspace_rejects_unknown_ccl() -> None:
    """
    Verify that an unknown CPU CCL strategy is rejected.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="Unsupported CPU CCL strategy"):
        create_cpu_workspace((5, 7), ccl="unknown")


def test_create_cpu_workspace_rejects_unavailable_cc3d() -> None:
    """
    Verify that requesting the unavailable optional cc3d binding fails clearly.

    Returns:
        None
    """
    with pytest.raises(RuntimeError, match="CC3D"):
        create_cpu_workspace((5, 7), ccl="cc3d")
