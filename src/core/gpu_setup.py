"""
src/core/gpu_setup.py
─────────────────────────────────────────────────────────────────────────────
Shared NVIDIA DLL registration for ONNX Runtime CUDA (Windows).

ONNX Runtime's CUDAExecutionProvider needs cublas64_12.dll, cublasLt64_12.dll,
cudnn64_9.dll etc. on the Windows DLL search path. PyTorch bundles all of
these in its own lib/ directory (torch.cuda already works), so we register
that directory first, then any pip-installed nvidia-* package bin dirs.

Must be called BEFORE the first CUDA InferenceSession is created. Provider
DLLs are loaded lazily on first session creation, so calling this at
component __init__ time is sufficient even if onnxruntime was already
imported.

Every entry point that creates an ONNX session (main.py, layer validators)
gets GPU support by calling register_nvidia_dlls() — without this, ONNX
Runtime silently falls back to CPU with only a warning on stderr.

Usage
-----
    from src.core.gpu_setup import register_nvidia_dlls, cuda_is_usable
    register_nvidia_dlls()   # idempotent, no-op on non-Windows
"""

import os

_registered = False


def register_nvidia_dlls():
    """
    Add PyTorch's bundled CUDA DLLs and pip nvidia-* bin dirs to the Windows
    DLL search path. Idempotent — safe to call from every component init.
    No-op on non-Windows platforms.
    """
    global _registered
    if _registered or os.name != "nt":
        _registered = True
        return

    dll_dirs = []

    # Strategy 1: PyTorch's bundled CUDA DLLs (most reliable — torch CUDA works)
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(torch_lib):
                dll_dirs.append(torch_lib)
    except Exception:
        pass

    # Strategy 2: pip-installed nvidia-cudnn-cu12 / nvidia-cublas-cu12 bin dirs
    try:
        import site
        for sp in site.getsitepackages():
            nvidia_base = os.path.join(sp, "nvidia")
            if not os.path.isdir(nvidia_base):
                continue
            for pkg in os.listdir(nvidia_base):
                bin_dir = os.path.join(nvidia_base, pkg, "bin")
                if os.path.isdir(bin_dir):
                    dll_dirs.append(bin_dir)
    except Exception:
        pass

    from src.core.logger import get_logger
    _log = get_logger("watcher.gpu")
    _log.debug(f"registering NVIDIA DLL dirs: {dll_dirs}")
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
            # Also prepend to PATH so implicit DLL loads by native code find them
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    _registered = True


def cuda_is_usable() -> bool:
    """
    True if ONNX Runtime can actually run a CUDA session — not just whether
    the provider is listed. get_available_providers() reports CUDA whenever
    onnxruntime-gpu is installed, even when the CUDA DLLs are missing and
    session creation would silently fall back to CPU. torch.cuda.is_available()
    is the reliable signal that the driver + DLLs are functional, since the
    provider DLLs we register come from torch's own lib directory.
    """
    register_nvidia_dlls()
    try:
        import torch
        import onnxruntime as ort
        return (
            torch.cuda.is_available()
            and "CUDAExecutionProvider" in ort.get_available_providers()
        )
    except Exception:
        return False
