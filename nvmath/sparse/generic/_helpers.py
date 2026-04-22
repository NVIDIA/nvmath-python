# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["compile_prolog", "compile_epilog", "compile_add", "compile_atomic_add", "compile_mul"]

from nvmath.sparse.ust._jit import compile_python_function

VALID_DTYPES = "float32", "float64", "double", "complex64", "complex128"


def compile_prolog(prolog, *, operand_label, dtype, compute_capability=None):
    """
    Compile a unary Python function with argument type ``dtype`` returning a value
    of type ``dtype`` to device code.

    Args:
        prolog: The unary Python function to compile to device code.
        operand_label: The operand to which the prolog applies ("a", "b").
        dtype: The dtype of the prolog's argument. Any scalar Numba type can be provided.
        compute_capability: The target compute capability, specified as a string
            (``'80'``, ``'89'``, ...). The default is the compute capability of the
            current device.

    Returns:
        The compiled device code as a bytes object.
    """

    valid_operand_labels = "a", "b", "c"
    if operand_label not in valid_operand_labels:
        raise ValueError(f"The operand label '{operand_label}' is invalid. It must one of {valid_operand_labels}.")

    # TODO: any type supported by numba-cuda can be provided.
    if dtype not in VALID_DTYPES:
        raise ValueError(f"The dtype '{dtype}' is invalid. It must one of {VALID_DTYPES}.")

    ltoir = compile_python_function(prolog, "prolog_" + operand_label, (dtype, dtype), compute_capability=compute_capability)
    return ltoir


def compile_epilog(epilog, *, dtype, compute_capability=None):
    """
    Compile a unary Python function with argument type ``dtype`` returning a value
    of type ``dtype`` to device code.

    Args:
        epilog: The unary Python function to compile to device code.
        dtype: The dtype of the epilog's argument. Any scalar Numba type can be provided.
        compute_capability: The target compute capability, specified as a string
            (``'80'``, ``'89'``, ...). The default is the compute capability of the
            current device.

    Returns:
        The compiled device code as a bytes object.
    """

    # TODO: any type supported by numba-cuda can be provided.
    if dtype not in VALID_DTYPES:
        raise ValueError(f"The dtype '{dtype}' is invalid. It must one of {VALID_DTYPES}.")

    ltoir = compile_python_function(epilog, "epilog", (dtype, dtype), compute_capability=compute_capability)
    return ltoir


def compile_add(add, *, dtype, compute_capability=None):
    """
    Compile a binary Python function with arguments of type ``dtype`` returning a value
    of type ``dtype`` to device code. This function must satisfy the mathematical
    requirements of a semiring addition operator.

    Args:
        add: The binary Python function to compile to device code.
        dtype: The dtype of the epilog's argument. Any scalar Numba type can be provided.
        compute_capability: The target compute capability, specified as a string
            (``'80'``, ``'89'``, ...). The default is the compute capability of the
            current device.

    Returns:
        The compiled device code as a bytes object.
    """

    # TODO: any type supported by numba-cuda can be provided.
    if dtype not in VALID_DTYPES:
        raise ValueError(f"The dtype '{dtype}' is invalid. It must one of {VALID_DTYPES}.")

    ltoir = compile_python_function(add, "add", (dtype, dtype, dtype), compute_capability=compute_capability)
    return ltoir


def compile_atomic_add(add, *, dtype, compute_capability=None):
    """
    Compile a binary Python function with arguments of type ``dtype`` returning a value
    of type ``dtype`` to device code. This function must satisfy the mathematical
    requirements of a semiring addition operator and must perform an **atomic** addition.

    Args:
        add: The binary Python function to compile to device code.
        dtype: The dtype of the epilog's argument. Any scalar Numba type can be provided.
        compute_capability: The target compute capability, specified as a string
            (``'80'``, ``'89'``, ...). The default is the compute capability of the
            current device.

    Returns:
        The compiled device code as a bytes object.
    """

    # TODO: any type supported by numba-cuda can be provided.
    if dtype not in VALID_DTYPES:
        raise ValueError(f"The dtype '{dtype}' is invalid. It must one of {VALID_DTYPES}.")

    ltoir = compile_python_function(add, "atomic_add", (dtype, f"{dtype}*", dtype), compute_capability=compute_capability)
    return ltoir


def compile_mul(add, *, dtype, compute_capability=None):
    """
    Compile a binary Python function with arguments of type ``dtype`` returning a value
    of type ``dtype`` to device code. This function must satisfy the mathematical
    requirements of a semiring multiplication operator.

    Args:
        add: The binary Python function to compile to device code.
        dtype: The dtype of the epilog's argument. Any scalar Numba type can be provided.
        compute_capability: The target compute capability, specified as a string
            (``'80'``, ``'89'``, ...). The default is the compute capability of the
            current device.

    Returns:
        The compiled device code as a bytes object.
    """

    # TODO: any type supported by numba-cuda can be provided.
    if dtype not in VALID_DTYPES:
        raise ValueError(f"The dtype '{dtype}' is invalid. It must one of {VALID_DTYPES}.")

    ltoir = compile_python_function(add, "mul", (dtype, dtype, dtype), compute_capability=compute_capability)
    return ltoir
