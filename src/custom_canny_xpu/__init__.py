"""Backend-aware Canny and RGB-to-grayscale operators."""

from ._cpu import CpuWorkspace, create_cpu_workspace

__all__ = ("CpuWorkspace", "create_cpu_workspace")
