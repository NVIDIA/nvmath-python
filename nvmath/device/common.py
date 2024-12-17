# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile

from .common_cuda import CodeType


SHARED_DEVICE_DOCSTRINGS = {
    "compiler": "A string to specify the compiler for the device code, currently supports ``None`` (default) and ``'Numba'``",
    #
    "precision": """\
The computation precision specified as a numpy float dtype, currently supports ``numpy.float16``, ``numpy.float32`` and
``numpy.float64``.""".replace("\n", " "),
    #
    "code_type": "The target GPU code and compute-capability.",
    #
    "execution": "A string specifying the execution method, can be ``'Block'`` or ``'Thread'``.",
}


# TODO: maybe pre-compile regular expression
def make_binary_tempfile(content, suffix):
    # TODO: may need to set it False for Windows? (refer to Python API doc)
    tmp = tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=True)
    tmp.write(content)
    tmp.flush()
    return tmp


def check_in(name, arg, set):
    if arg not in set:
        raise ValueError(f"{name} must be in {set} ; got {name} = {arg}")


def check_not_in(name, arg, set):
    if arg in set:
        raise ValueError(f"{name} must not be any of those value {set} ; got {name} = {arg}")


def check_contains(set, key):
    if key not in set:
        raise ValueError(f"{key} must be in {set}")


def check_dim3(name, arg):
    if len(arg) != 3:
        raise ValueError(f"{name} should be a length-3 tuple ; got {name} = {arg}")


def check_code_type(code_type):
    if isinstance(code_type, CodeType):
        if code_type.cc.major < 7:
            raise ValueError(f"code_type.cc.major must be >= 7 ; got code_type.cc.major = {code_type.cc.major}")
        if code_type.cc.minor < 0:
            raise ValueError(f"code_type.cc.minor must be >= 0 ; got code_type.cc.minor = {code_type.cc.minor}")
        check_in("code_type.kind", code_type.kind, ["lto"])
    else:
        raise ValueError(f"code_type should be an instance of CodeType ; got code_type = {code_type}")


def find_unsigned(name, txt):
    regex = re.compile(f".global .align 4 .u32 {name} = ([0-9]*);", re.MULTILINE)
    found = regex.search(txt)
    if found is None:  # TODO: improve regex logic
        regex = re.compile(f".global .align 4 .u32 {name};", re.MULTILINE)
        found = regex.search(txt)
        if found is not None:
            return 0
        else:
            raise ValueError(f"{name} not found in text")
    else:
        return int(found.group(1))


def find_mangled_name(name, txt):
    regex = re.compile(f"[_a-zA-Z0-9]*{name}[_a-zA-Z0-9]*", re.MULTILINE)
    return regex.search(txt).group(0)


def find_dim2(name, txt):
    return (find_unsigned(f"{name}_x", txt), find_unsigned(f"{name}_y", txt))


def find_dim3(name, txt):
    return (find_unsigned(f"{name}_x", txt), find_unsigned(f"{name}_y", txt), find_unsigned(f"{name}_z", txt))
