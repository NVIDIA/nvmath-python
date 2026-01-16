# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import re
import subprocess
import sys

try:
    from cuda.core import Device, system
    from cuda.core._utils.cuda_utils import CUDAError
except ImportError:
    from cuda.core.experimental import Device, system
    from cuda.core.experimental._utils.cuda_utils import CUDAError

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    # disable plot windows from popping out when testing locally
    matplotlib.use("Agg")
import pytest

try:
    try:
        num_devices = system.get_num_devices()
    except AttributeError:
        num_devices = system.num_devices
    DEVICE_COUNT = num_devices
except CUDAError:
    DEVICE_COUNT = 0

try:
    cc = tuple(Device().compute_capability)
except RuntimeError:
    cc = (0, 0)


class SampleTestError(Exception):
    pass


def parse_python_script(filepath):
    if filepath.endswith(".py"):
        with open(filepath, encoding="utf-8") as f:
            script = f.read()
    else:
        raise ValueError(f"{filepath} not supported")
    return script


def run_sample(samples_path, filename, env=None, use_subprocess=False, use_mpi=False):
    requires_mgpu = filename.endswith("_mgpu.py")
    if DEVICE_COUNT == 0 and "cpu_execution" not in filename:
        pytest.skip(f"Sample ({filename}) skipped because 0 CUDA devices were found.")
    if requires_mgpu and DEVICE_COUNT == 1:
        pytest.skip(f"Sample ({filename}) skipped due to limited device counts : ({DEVICE_COUNT})")
    fullpath = os.path.join(samples_path, filename)
    script = parse_python_script(fullpath)
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        SYS_PATH_BACKUP = sys.path.copy()
        sys.path.append(samples_path)
        if use_mpi:
            assert use_subprocess
            # Check if the filename indicates with how many processes to run, for example:
            # `example_something_4p.py` is to be run with 4 processes.
            m = re.search(r".*_(\d+)p.py$", filename)
            uses_nccl = "distributed/linalg/advanced/matmul" in fullpath
            if m:
                num_procs = m.group(1)
                if uses_nccl and int(num_procs) > DEVICE_COUNT:
                    pytest.skip(
                        f"This test requires {num_procs} processes but NCCL only allows one "
                        f"process per GPU and there are {DEVICE_COUNT} GPUs"
                    )
            elif uses_nccl:
                # NCCL only allows one process per GPU.
                num_procs = str(DEVICE_COUNT)
            else:
                # Run with 2 processes by default.
                num_procs = "2"
            cmd = ["mpiexec", "-n", num_procs, sys.executable, fullpath]
        else:
            cmd = [sys.executable, fullpath]
        if use_subprocess:
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
            if result.returncode != 0:
                if "ModuleNotFoundError" in result.stderr:
                    raise ModuleNotFoundError(result.stderr)
                else:
                    raise RuntimeError(f"Subprocess failed: {result.stderr}")
        else:
            exec(script, env if env is not None else {})
    except ImportError as e:
        # for samples requiring any of optional dependencies
        for m in ("torch", "cupy"):
            if f"No module named '{m}'" in str(e):
                pytest.skip(f"{m} uninstalled, skipping related tests")
                break
        else:
            raise
    except Exception as e:
        if str(e) == "libcudadevrt.a not found":
            pytest.skip(f"Skipping test {filename} since libcudadevrt.a is not found.")
        else:
            msg = "\n"
            msg += f"Got error ({filename}):\n"
            msg += str(e)
            raise SampleTestError(msg) from e
    finally:
        sys.path = SYS_PATH_BACKUP
        sys.argv = old_argv
        # further reduce the memory watermark
        gc.collect()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
