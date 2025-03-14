# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import operator

from nvmath.device import curand_kernel
from nvmath.device.common_mathdx import CURAND_HOME  # noqa: F401

from numba import cuda, types
from numba.extending import models, register_model, typeof_impl
from numba.cuda.cudadecl import register_global
from numba.cuda.cudaimpl import lower
from numba.core.typing.templates import AbstractTemplate, signature
import numpy as np

from llvmlite import ir

xorwow_dtype = np.dtype(
    [
        ("d", np.uint32),
        ("v", np.uint32, (5,)),
        ("boxmuller_flag", np.int32),
        ("boxmuller_flag_double", np.int32),
        ("boxmuller_extra", np.float32),
        ("boxmuller_extra_double", np.float64),
    ],
    align=True,
)
test_dtype = np.dtype([("v", np.uint32)], align=True)
mrg32k3a_dtype = np.dtype(
    [
        ("s1", np.uint32, (3,)),
        ("s2", np.uint32, (3,)),
        ("boxmuller_flag", np.int32),
        ("boxmuller_flag_double", np.int32),
        ("boxmuller_extra", np.float32),
        ("boxmuller_extra_double", np.float64),
    ],
    align=True,
)
sobol32_dtype = np.dtype(
    [("i", np.uint32), ("x", np.uint32), ("c", np.uint32), ("direction_vectors", np.int32, (32,))], align=True
)
scrambled_sobol32_dtype = np.dtype(
    [("i", np.uint32), ("x", np.uint32), ("c", np.uint32), ("direction_vectors", np.int32, (32,))], align=True
)
sobol64_dtype = np.dtype(
    [("i", np.uint64), ("x", np.uint64), ("c", np.uint64), ("direction_vectors", np.int64, (64,))], align=True
)
scrambled_sobol64_dtype = np.dtype(
    [("i", np.uint64), ("x", np.uint64), ("c", np.uint64), ("direction_vectors", np.int64, (64,))], align=True
)
philox4_32_10_dtype = np.dtype(
    [
        ("ctr", np.uint32, (4,)),
        ("output", np.uint32, (4,)),
        ("key", np.uint32, (2,)),
        ("STATE", np.uint32),
        ("boxmuller_flag", np.int32),
        ("boxmuller_flag_double", np.int32),
        ("boxmuller_extra", np.float32),
        ("boxmuller_extra_double", np.float64),
    ],
    align=True,
)

NP_DTYPES = {
    "curandStateXORWOW": xorwow_dtype,
    "curandStateTest": test_dtype,
    "curandStateMRG32k3a": mrg32k3a_dtype,
    "curandStateSobol32": sobol32_dtype,
    "curandStateScrambledSobol32": scrambled_sobol32_dtype,
    "curandStateSobol64": sobol64_dtype,
    "curandStateScrambledSobol64": scrambled_sobol64_dtype,
    "curandStatePhilox4_32_10": philox4_32_10_dtype,
}


def make_curand_states(name: str):
    """Create a boxed curand states object to represent an array of curand state.
    Should invoke after creating the numba types for each curand state.
    """

    curand_state_name = name

    state_ty = getattr(curand_kernel, f"_type_{curand_state_name}")

    curand_states_name = curand_state_name.replace("State", "States")

    # cuRAND state type as a NumPy dtype - this mirrors the state defined in
    # curand_kernel.h. Can be used to inspect the state through the device array
    # held by CurandStates.

    curandStateDtype = NP_DTYPES[curand_state_name]

    # Hold an array of cuRAND states - somewhat analogous to a curandState* in
    # C/C++.

    class CurandStates:
        def __init__(self, n):
            self._array = cuda.device_array(n, dtype=curandStateDtype)

        @property
        def data(self):
            return self._array.__cuda_array_interface__["data"][0]

    CurandStates.__name__ = curand_states_name

    class CurandStatePointer(types.Type):
        def __init__(self, name):
            self.dtype = state_ty
            super().__init__(name=name)

    curand_state_pointer = CurandStatePointer(curand_states_name + "*")

    @typeof_impl.register(CurandStates)
    def typeof_curand_states(val, c):
        return curand_state_pointer

    @register_global(operator.getitem)
    class CurandStatesPointerGetItem(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            [ptr, idx] = args
            if ptr == curand_state_pointer:
                return signature(types.CPointer(state_ty), curand_state_pointer, types.int64)

    register_model(CurandStatePointer)(models.PointerModel)

    @lower(operator.getitem, curand_state_pointer, types.int64)
    def lower_curand_states_getitem(context, builder, sig, args):
        [ptr, idx] = args

        # Working out my own GEP
        ptrint = builder.ptrtoint(ptr, ir.IntType(64))
        itemsize = curandStateDtype.itemsize
        offset = builder.mul(idx, context.get_constant(types.int64, itemsize))
        ptrint = builder.add(ptrint, offset)
        ptr = builder.inttoptr(ptrint, ptr.type)
        return ptr

    # Argument handling. When a CurandStatePointer is passed into a kernel, we
    # really only need to pass the pointer to the data, not the whole underlying
    # array structure. Our handler here transforms these arguments into a uint64
    # holding the pointer.

    class CurandStateArgHandler:
        def prepare_args(self, ty, val, **kwargs):
            if isinstance(val, CurandStates):
                assert ty == curand_state_pointer
                return types.uint64, val.data
            else:
                return ty, val

    curand_state_arg_handler = CurandStateArgHandler()

    return CurandStates, curand_state_arg_handler


curandStates = [
    "curandStateTest",
    "curandStateXORWOW",
    "curandStateMRG32k3a",
    "curandStateSobol32",
    "curandStateScrambledSobol32",
    "curandStateSobol64",
    "curandStateScrambledSobol64",
    "curandStatePhilox4_32_10",
    # Require additional type parsing of mtgp32_params_fast and mtgp32_kernel_params
    # "curandStateMtgp32",
]

numpy_curand_states = []
states_arg_handlers = []

for state in curandStates:
    states_obj, arg_handler = make_curand_states(state)
    numpy_curand_states.append(states_obj)
    states_arg_handlers.append(arg_handler)


# Export
globals().update({s.__name__: s for s in numpy_curand_states})
globals().update({"states_arg_handlers": states_arg_handlers})

__all__ = list(  # noqa: F822
    {s.__name__ for s in numpy_curand_states} | {"states_arg_handlers"}
)
