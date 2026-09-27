#!/usr/bin/env python3
"""Stage 00: verify the box is ready before anything else runs.

Checks Python/torch/CUDA versions, exercises a bf16 matmul on every visible
GPU (to catch a GPU whose kernels don't actually work, e.g. a too-new
compute capability the installed torch build doesn't support), and checks
free RAM, free disk, CPU core count and Hugging Face reachability.

Exits non-zero on any hard failure (missing torch, no working CUDA matmul,
disk/RAM below the minimum). Hugging Face unreachable is a warning only,
since a box with pre-downloaded models can still run offline.
"""
from __future__ import annotations

import argparse
import multiprocessing
import shutil
import subprocess
import sys
import urllib.request

MIN_FREE_RAM_GB = 8
MIN_FREE_DISK_GB = 20


def check_python() -> None:
    print(f"python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        fail("python >= 3.10 required")


def check_torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        fail("torch is not installed")
        return False

    print(f"torch: {torch.__version__}")
    if not torch.cuda.is_available():
        if nvidia_gpu_present():
            # Seen after a crash left the driver needing a reset (kernel log:
            # "Xid 154 ... GPU Reset Required"): without this check the GPU
            # stages would silently fall back to CPU and take days.
            fail("an NVIDIA GPU is installed but CUDA is unusable (driver needs a reset?) — "
                 "reboot, check `nvidia-smi`, then rerun")
        else:
            warn("CUDA not available — GPU stages (03/07/08) will run on CPU or be skipped")
        return False

    print(f"cuda: {torch.version.cuda}")
    ok = True
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"gpu[{i}]: {name} (sm_{cap[0]}{cap[1]})")
        try:
            a = torch.randn(1024, 1024, device=i, dtype=torch.bfloat16)
            b = torch.randn(1024, 1024, device=i, dtype=torch.bfloat16)
            c = a @ b
            torch.cuda.synchronize(i)
            assert torch.isfinite(c).all()
            print(f"gpu[{i}]: bf16 matmul OK")
        except Exception as e:  # noqa: BLE001 - report and keep checking other GPUs
            fail(f"gpu[{i}]: bf16 matmul failed: {e}")
            ok = False
    return ok


def nvidia_gpu_present() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "GPU" in out.stdout


def check_resources(min_ram_gb: float, min_disk_gb: float) -> None:
    print(f"cpu cores: {multiprocessing.cpu_count()}")

    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        avail_kb = int([l for l in meminfo.splitlines() if l.startswith("MemAvailable")][0].split()[1])
        avail_gb = avail_kb / 1024 / 1024
    except Exception:
        avail_gb = None

    if avail_gb is not None:
        print(f"free ram: {avail_gb:.1f} GB")
        if avail_gb < min_ram_gb:
            fail(f"free RAM {avail_gb:.1f} GB < required {min_ram_gb} GB")
    else:
        warn("could not read /proc/meminfo")

    free_gb = shutil.disk_usage(".").free / 1024**3
    print(f"free disk: {free_gb:.1f} GB")
    if free_gb < min_disk_gb:
        fail(f"free disk {free_gb:.1f} GB < required {min_disk_gb} GB")


def check_hf_reachable() -> None:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        print("huggingface.co: reachable")
    except Exception as e:  # noqa: BLE001
        warn(f"huggingface.co unreachable ({e}) — models must already be cached locally")


_failed = False


def fail(msg: str) -> None:
    global _failed
    _failed = True
    print(f"FAIL: {msg}")


def warn(msg: str) -> None:
    print(f"WARN: {msg}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-ram-gb", type=float, default=MIN_FREE_RAM_GB, help="lower this for a small local smoke test")
    p.add_argument("--min-disk-gb", type=float, default=MIN_FREE_DISK_GB)
    args, _ = p.parse_known_args()
    check_python()
    check_torch_cuda()
    check_resources(args.min_ram_gb, args.min_disk_gb)
    check_hf_reachable()
    if _failed:
        print("00_check_env: FAIL")
        return 1
    print("00_check_env: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
