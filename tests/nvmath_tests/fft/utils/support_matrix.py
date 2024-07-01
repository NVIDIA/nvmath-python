# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .common_axes import Framework, Backend, DType, ShapeKind, Direction, OptFftType


framework_backend_support = {
    Framework.cupy: [Backend.gpu],
    Framework.numpy: [Backend.cpu],
    Framework.torch: [Backend.cpu, Backend.gpu],
}

framework_type_support = {
    Framework.cupy: [
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.numpy: [
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.torch: [
        DType.float16,
        DType.float32,
        DType.float64,
        DType.complex32,
        DType.complex64,
        DType.complex128,
    ],
}

type_shape_support = {
    # Note, there is an undocumented partial support for prime shapes above 128
    DType.float16: [ShapeKind.pow2],
    DType.float32: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    DType.float64: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    # Note, there is an undocumented partial support for prime shapes above 128
    DType.complex32: [ShapeKind.pow2],
    DType.complex64: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    DType.complex128: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
}

opt_fft_type_direction_support = {
    OptFftType.c2c: [Direction.forward, Direction.inverse],
    OptFftType.r2c: [Direction.forward],
    OptFftType.c2r: [Direction.inverse],
}

opt_fft_type_input_type_support = {
    OptFftType.c2c: [DType.complex32, DType.complex64, DType.complex128],
    OptFftType.r2c: [DType.float16, DType.float32, DType.float64],
    OptFftType.c2r: [DType.complex32, DType.complex64, DType.complex128],
}

inplace_opt_ftt_type_support = {True: [OptFftType.c2c], False: list(OptFftType)}
