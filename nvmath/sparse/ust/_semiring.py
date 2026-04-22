# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the default semiring operations.
"""

__all__ = []

semiring_ops_decl = """
extern "C" __device__ CTP add(CTP, CTP);
extern "C" __device__ ATP atomic_add(ATP *, ATP);
extern "C" __device__ CTP mul(CTP, CTP);
"""

semiring_add_impl = """
extern "C" __device__
CTP add(CTP a, CTP b) {
    return a + b;
}
"""

semiring_atomic_add_impl = """
extern "C" __device__
ATP atomic_add(ATP *a, ATP b) {
    return atomicAdd(a, b);
}
"""

semiring_mul_impl = """
extern "C" __device__
CTP mul(CTP a, CTP b) {
    return a * b;
}
"""

prolog_epilog_decl = """
extern "C" __device__ CTP prolog_a(CTP);
extern "C" __device__ CTP prolog_b(CTP);
extern "C" __device__ CTP prolog_c(CTP);
extern "C" __device__ CTP epilog(CTP);
"""

prolog_a_impl = """
extern "C" __device__
CTP prolog_a(CTP a) {
    return a;
}
"""

prolog_b_impl = """
extern "C" __device__
CTP prolog_b(CTP b) {
    return b;
}
"""

prolog_c_impl = """
extern "C" __device__
CTP prolog_c(CTP c) {
    return c;
}
"""

epilog_impl = """
extern "C" __device__
CTP epilog(CTP c) {
    return c;
}
"""
