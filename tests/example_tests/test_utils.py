# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import sys

import cupy as cp
try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    # disable plot windows from popping out when testing locally
    matplotlib.use('Agg')
import pytest

DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()

class SampleTestError(Exception):
    pass


def parse_python_script(filepath):
    if filepath.endswith('.py'):
        with open(filepath, "r", encoding='utf-8') as f:
            script = f.read()
    else:
        raise ValueError(f"{filepath} not supported")
    return script


def run_sample(samples_path, filename, env=None):
    requires_mgpu = filename.endswith('_mgpu.py')
    if DEVICE_COUNT == 0:
        raise SystemError("No active device found")
    if requires_mgpu and DEVICE_COUNT == 1:
        pytest.skip(f"Sample ({filename}) skipped due to limited device counts : ({DEVICE_COUNT})")
    fullpath = os.path.join(samples_path, filename)
    script = parse_python_script(fullpath)
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        SYS_PATH_BACKUP = sys.path.copy()
        sys.path.append(samples_path)
        exec(script, env if env is not None else {})
    except ImportError as e:
        # for samples requiring any of optional dependencies
        for m in ('torch',):
            if f"No module named '{m}'" in str(e):
                pytest.skip(f'{m} uninstalled, skipping related tests')
                break
        else:
            raise
    except Exception as e:
        if str(e) == "libcudadevrt.a not found":
            pytest.skip(f'Skipping test {filename} since libcudadevrt.a is not found.')
        else:
            msg = "\n"
            msg += f'Got error ({filename}):\n'
            msg += str(e)
            raise SampleTestError(msg) from e
    finally:
        sys.path = SYS_PATH_BACKUP
        sys.argv = old_argv
        # further reduce the memory watermark
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
