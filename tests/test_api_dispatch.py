"""Tests for public-boundary backend resolution and CPU dispatch."""

# Pure backend-resolution helpers remain directly tested while GPU execution
# is absent; this isolates the selection table without mocking the whole public
# operator. Reassess these tests when the GPU public path is integrated.
# pylint: disable=protected-access

import sys
from types import ModuleType
from typing import Any, cast

import numpy as npy

# Pyright environment discovery misses the installed pytest package.
import pytest  # pyright: ignore[reportMissingImports]

from custom_canny_xpu import _api, _cpu
from custom_canny_xpu._cpu import BoolArray2D, CpuWorkspace, Float32Array2D


@pytest.mark.parametrize(
    "backend",
    (_api.CannyBackend.AUTO, _api.CannyBackend.CPU, "auto", "cpu"),
)
def test_resolve_canny_backend_accepts_cpu_selection(
    backend: _api.CannyBackend | str,
) -> None:
    """
    Verify automatic and explicit CPU selection accept a NumPy input.

    Args:
        backend: automatic or explicit CPU backend representation
    Returns:
        None.
    """
    image = npy.zeros((2, 2), dtype=npy.float32)

    assert _api._resolve_canny_backend(image, backend) is _api.CannyBackend.CPU


@pytest.mark.parametrize("backend", ("invalid", 1, None))
def test_resolve_canny_backend_rejects_invalid_names(backend: Any) -> None:
    """
    Verify unsupported backend names are rejected before residency dispatch.

    Args:
        backend: unsupported backend selector
    Returns:
        None.
    """
    image = npy.zeros((2, 2), dtype=npy.float32)

    with pytest.raises(ValueError, match="Unsupported backend"):
        _api._resolve_canny_backend(image, backend)


def test_resolve_canny_backend_rejects_cpu_gpu_residency_mismatch() -> None:
    """
    Verify explicit GPU execution rejects a CPU-resident NumPy input.

    Returns:
        None.
    """
    image = npy.zeros((2, 2), dtype=npy.float32)

    with pytest.raises(TypeError, match="cannot process"):
        _api._resolve_canny_backend(image, "gpu")


def test_resolve_canny_backend_rejects_unknown_array_type() -> None:
    """
    Verify backend resolution rejects objects from no supported array backend.

    Returns:
        None.
    """
    with pytest.raises(TypeError, match="NumPy or CuPy"):
        _api._resolve_canny_backend([[0.0]], "auto")


def test_detect_array_backend_recognizes_loaded_cupy_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify residency detection recognizes CuPy arrays without importing CuPy.

    Args:
        monkeypatch: pytest helper used to install a temporary CuPy module double
    Returns:
        None.
    """
    class FakeCupyArray:  # pylint: disable=too-few-public-methods # Marker type for backend detection
        """Stand in for the CuPy ndarray type without requiring CuPy."""

    fake_cupy_module = ModuleType("cupy")
    fake_cupy_module.__dict__["ndarray"] = FakeCupyArray
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_module)

    assert (
        _api._detect_array_backend(FakeCupyArray()) is _api.CannyBackend.GPU
    )


def test_resolve_canny_backend_reports_unavailable_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify a GPU-resident input fails clearly while GPU integration is absent.

    Args:
        monkeypatch: pytest helper used to isolate array-residency detection
    Returns:
        None.
    """
    def detect_gpu_array_backend(image: Any) -> _api.CannyBackend:
        """
        Return GPU residency for the resolver test input.

        Args:
            image: ignored test object
        Returns:
            GPU backend marker
        """
        _ = image
        return _api.CannyBackend.GPU

    monkeypatch.setattr(_api, "_detect_array_backend", detect_gpu_array_backend)

    with pytest.raises(RuntimeError, match="GPU backend is unavailable"):
        _api._resolve_canny_backend(object(), "auto")


def test_canny_dispatches_valid_cpu_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify Canny validates and dispatches a valid call through the CPU boundary.

    Args:
        monkeypatch: pytest helper used to replace unfinished numerical execution
    Returns:
        None.
    """
    image = npy.zeros((12, 13), dtype=npy.float32)
    expected_output = cast(
        BoolArray2D, npy.zeros_like(image, dtype=npy.bool_)
    )
    received: dict[str, Any] = {}

    def execute_cpu_canny(  # pylint: disable=too-many-arguments # Test double mirrors the CPU executor
        dispatched_image: Float32Array2D,
        sigma: float,
        gaussian_radius: int,
        low_threshold: float,
        high_threshold: float,
        *,
        workspace: CpuWorkspace,
    ) -> BoolArray2D:
        """
        Record the validated CPU dispatch and return a deterministic edge map.

        Args:
            dispatched_image: validated image supplied to CPU execution
            sigma: validated Gaussian standard deviation
            gaussian_radius: Gaussian radius derived by the API
            low_threshold: validated weak-edge threshold
            high_threshold: validated strong-edge threshold
            workspace: prepared CPU workspace
        Returns:
            deterministic Boolean edge map
        """
        received.update(
            image=dispatched_image,
            sigma=sigma,
            gaussian_radius=gaussian_radius,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            workspace=workspace,
        )
        return expected_output

    monkeypatch.setattr(_cpu, "_execute_cpu_canny", execute_cpu_canny)

    result = _api.canny(image)

    assert result is expected_output
    assert received["image"] is image
    assert received["sigma"] == 1.5
    assert received["gaussian_radius"] == 6
    assert received["low_threshold"] == 0.1
    assert received["high_threshold"] == 0.2
    assert isinstance(received["workspace"], CpuWorkspace)


def test_canny_reuses_supplied_cpu_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify public dispatch borrows a compatible caller-supplied CPU workspace.

    Args:
        monkeypatch: pytest helper used to replace unfinished numerical execution
    Returns:
        None.
    """
    image = npy.zeros((12, 13), dtype=npy.float32)
    workspace = _cpu.create_cpu_workspace((image.shape[0], image.shape[1]))

    def execute_cpu_canny(  # pylint: disable=too-many-arguments # Test double mirrors the CPU executor
        dispatched_image: Float32Array2D,
        sigma: float,
        gaussian_radius: int,
        low_threshold: float,
        high_threshold: float,
        *,
        workspace: CpuWorkspace,
    ) -> BoolArray2D:
        """
        Assert workspace identity and return a deterministic edge map.

        Args:
            dispatched_image: validated image supplied to CPU execution
            sigma: validated Gaussian standard deviation
            gaussian_radius: Gaussian radius derived by the API
            low_threshold: validated weak-edge threshold
            high_threshold: validated strong-edge threshold
            workspace: prepared CPU workspace
        Returns:
            deterministic Boolean edge map
        """
        _ = (sigma, gaussian_radius, low_threshold, high_threshold)
        assert workspace is supplied_workspace
        return cast(
            BoolArray2D,
            npy.zeros_like(dispatched_image, dtype=npy.bool_),
        )

    supplied_workspace = workspace
    monkeypatch.setattr(_cpu, "_execute_cpu_canny", execute_cpu_canny)

    result = _api.canny(image, workspace=workspace)

    assert result.shape == image.shape
    assert result.dtype == npy.bool_


def test_canny_rejects_invalid_image_before_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify invalid CPU dimensions fail before temporary workspace construction.

    Args:
        monkeypatch: pytest helper used to detect forbidden workspace allocation
    Returns:
        None.
    """
    image = npy.zeros((11, 12), dtype=npy.float32)

    def fail_if_workspace_is_created(
        shape: tuple[int, int], *, ccl: Any = None
    ) -> CpuWorkspace:
        """
        Fail if invalid input reaches workspace construction.

        Args:
            shape: requested workspace shape
            ccl: requested CCL strategy
        Returns:
            no workspace because this function must not be called
        Raises:
            AssertionError: whenever called
        """
        raise AssertionError(f"Unexpected workspace allocation for {shape}, {ccl}")

    monkeypatch.setattr(_cpu, "create_cpu_workspace", fail_if_workspace_is_created)

    with pytest.raises(ValueError, match=r"received shape \(11, 12\)"):
        _api.canny(image)


def test_canny_rejects_explicit_ccl_with_workspace() -> None:
    """
    Verify the public CPU path rejects simultaneous CCL and workspace selection.

    Returns:
        None.
    """
    image = npy.zeros((12, 12), dtype=npy.float32)
    workspace = _cpu.create_cpu_workspace((image.shape[0], image.shape[1]))

    with pytest.raises(ValueError, match="cannot both"):
        _api.canny(image, ccl="scipy", workspace=workspace)
