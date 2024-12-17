# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["LeadingDimension", "TransposeMode"]

from collections import namedtuple

from .caching import json_hash
from .common import check_in, check_code_type
from .common_cpp import generate_type_map, NP_TYPES_TO_CPP_TYPES
from .common_cuda import Dim3
from .types import REAL_NP_TYPES


class LeadingDimension(namedtuple("LeadingDimension", ["a", "b", "c"])):
    """
    A namedtuple class that encapsulates the three leading dimensions in matrix
    multiplication :math:`C = \\alpha Op(A) Op(B) + \\beta C`.

    Attributes:
        a (int): The leading dimension of two-dimensional array used to store the matrix
            ``A``.

        b (int): The leading dimension of two-dimensional array used to store the matrix
            ``B``.

        c (int): The leading dimension of two-dimensional array used to store the matrix
            ``C``.
    """

    pass


class TransposeMode(namedtuple("TransposeMode", ["a", "b"])):
    """
    A namedtuple class that encapsulates the transpose mode for input matrices ``A`` and
    ``B`` in matrix multiplication.

    Attributes:
        a: The operation that needs to be performed with input matrix ``A``, currently
            supports ``'non_transposed'``, ``'transposed'`` and ``'conj_transposed'``.

        b: The operation that needs to be performed with input matrix ``B``, currently
            supports ``'non_transposed'``, ``'transposed'`` and ``'conj_transposed'``.
    """

    pass


def validate(size, data_type, precision, execution, transpose_mode, block_dim, code_type, function, leading_dimension):
    (m, n, k) = size
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be > 0. Got {m}, {n}, {k}")
    check_in("precision", precision, REAL_NP_TYPES)
    check_in("data_type", data_type, ["real", "complex"])
    check_in("execution", execution, ["Block", "Thread"])
    check_in("function", function, ["MM"])
    if transpose_mode is not None:
        allowed_values = ["non_transposed", "transposed", "conj_transposed"]
        if isinstance(transpose_mode, TransposeMode):
            check_in("transpose_mode.a", transpose_mode.a, allowed_values)
            check_in("transpose_mode.b", transpose_mode.b, allowed_values)
        else:
            raise ValueError(
                f"transpose_mode should be an instance of {TransposeMode} or a 2-tuple, and individual fields "
                f"should be one of {allowed_values}. Instead got transpose_mode = {transpose_mode}"
            )
    if block_dim in (None, "suggested"):
        pass
    elif isinstance(block_dim, Dim3):
        prod = block_dim[0] * block_dim[1] * block_dim[2]
        if prod <= 0 or prod > 1024:
            raise ValueError(
                f"The product of the entries in block_dim should be between 1 and 1024 ; got block_dim = {block_dim}"
            )
    else:
        raise ValueError(f"block_dim should be None, a Dim3 instance or 'suggested'; got block_dim = {block_dim}")
    if code_type is not None:
        check_code_type(code_type)
    if leading_dimension in (None, "suggested") or isinstance(leading_dimension, LeadingDimension):
        pass
    else:
        raise ValueError(
            f"leading_dimension should be None, a LeadingDimension instance or 'suggested'; "
            f"got leading_dimension = {leading_dimension}"
        )


def generate_MM(size, precision, data_type, function, transpose_mode, code_type, block_dim, execution, leading_dimension):
    if block_dim is not None:
        block_dim = f"+ BlockDim<{ block_dim[0] }, { block_dim[1] }, { block_dim[2] }>()"
    else:
        block_dim = ""

    if code_type is not None:
        sm = f"+ SM<{ code_type.cc.major * 100 + code_type.cc.minor * 10 }>()"
    else:
        sm = ""

    if leading_dimension is not None:
        leading_dimension = f"+ LeadingDimension<{ leading_dimension.a }, { leading_dimension.b }, { leading_dimension.c }>()"
    else:
        leading_dimension = ""

    if transpose_mode is not None:
        trans = f"+ TransposeMode<transpose_mode::{ transpose_mode.a }, transpose_mode::{ transpose_mode.b }>()"
    else:
        trans = ""

    (m, n, k) = size

    cpp = f"""\
    using MM  = decltype(   Size<{ m }, { n }, { k }>()
                          + Precision<{ NP_TYPES_TO_CPP_TYPES[precision] }>()
                          + Type<type::{ data_type }>()
                          + Function<function::{ function }>()
                          + { execution }()
                          { trans }
                          { sm }
                          { block_dim }
                          { leading_dimension }
                        );
    """

    hash = json_hash(
        size=size,
        precision=NP_TYPES_TO_CPP_TYPES[precision],
        data_type=data_type,
        function=function,
        trans=trans,
        code_type=code_type,
        block_dim=block_dim,
        execution=execution,
    )

    return cpp, hash


def generate_block(size, precision, data_type, function, transpose_mode, code_type, block_dim, execution, leading_dimension):
    MM, name = generate_MM(
        size=size,
        precision=precision,
        data_type=data_type,
        function=function,
        transpose_mode=transpose_mode,
        code_type=code_type,
        block_dim=block_dim,
        leading_dimension=leading_dimension,
        execution=execution,
    )
    api_name_basic = f"libmathdx_function_matmul_base_{name}"
    api_name_ldabc = f"libmathdx_function_matmul_ldabc_{name}"

    TYPE_MAP, type_map_name = generate_type_map(name=name)

    cpp = f"""\
    #include <cublasdx.hpp>
    using namespace cublasdx;

    { MM }

    { TYPE_MAP }

    __device__ constexpr unsigned  value_type = {type_map_name}<MM::value_type>::value;
    __device__ constexpr unsigned  input_type = {type_map_name}<MM::input_type>::value;
    __device__ constexpr unsigned  output_type = {type_map_name}<MM::output_type>::value;

    __device__ constexpr unsigned a_dim_x = cuda::std::get<0>(MM::a_dim);
    __device__ constexpr unsigned a_dim_y = cuda::std::get<1>(MM::a_dim);

    __device__ constexpr unsigned b_dim_x = cuda::std::get<0>(MM::b_dim);
    __device__ constexpr unsigned b_dim_y = cuda::std::get<1>(MM::b_dim);

    __device__ constexpr unsigned c_dim_x = cuda::std::get<0>(MM::c_dim);
    __device__ constexpr unsigned c_dim_y = cuda::std::get<1>(MM::c_dim);

    __device__ constexpr unsigned lda = MM::lda;
    __device__ constexpr unsigned ldb = MM::ldb;
    __device__ constexpr unsigned ldc = MM::ldc;

    __device__ constexpr unsigned a_size = MM::a_size;
    __device__ constexpr unsigned b_size = MM::b_size;
    __device__ constexpr unsigned c_size = MM::c_size;

    __device__ constexpr unsigned shared_memory_size = MM::shared_memory_size;

    __device__ constexpr unsigned block_dim_x = MM::block_dim.x;
    __device__ constexpr unsigned block_dim_y = MM::block_dim.y;
    __device__ constexpr unsigned block_dim_z = MM::block_dim.z;

    __device__ constexpr unsigned suggested_block_dim_x = MM::suggested_block_dim.x;
    __device__ constexpr unsigned suggested_block_dim_y = MM::suggested_block_dim.y;
    __device__ constexpr unsigned suggested_block_dim_z = MM::suggested_block_dim.z;

    __device__ constexpr unsigned max_threads_per_block = MM::max_threads_per_block;

    using MMTransposeMode = transpose_mode_of<MM>;
    __device__ constexpr int a_trans = (int)MMTransposeMode::a_transpose_mode;
    __device__ constexpr int b_trans = (int)MMTransposeMode::b_transpose_mode;

    __device__ void { api_name_basic }(MM::input_type*  alpha,
                                       MM::input_type*  smem_a,
                                       MM::input_type*  smem_b,
                                       MM::output_type* beta,
                                       MM::output_type* smem_c) {{
        MM().execute(*alpha, smem_a, smem_b, *beta, smem_c);
    }}

    __device__ void { api_name_ldabc }(MM::input_type*    alpha,
                                       MM::input_type*    smem_a,
                                       const unsigned int lda,
                                       MM::input_type*    smem_b,
                                       const unsigned int ldb,
                                       MM::output_type*   beta,
                                       MM::output_type*   smem_c,
                                       const unsigned int ldc) {{
        MM().execute(*alpha, smem_a, lda, smem_b, ldb, *beta, smem_c, ldc);
    }}
    """

    return {"cpp": cpp, "names": {"smem_basic": api_name_basic, "smem_ldabc": api_name_ldabc}}


def generate_block_ld(size, precision, data_type, function, transpose_mode, code_type, block_dim, leading_dimension, execution):
    MM_str, _ = generate_MM(
        size=size,
        precision=precision,
        data_type=data_type,
        function=function,
        transpose_mode=transpose_mode,
        code_type=code_type,
        block_dim=block_dim,
        leading_dimension=leading_dimension,
        execution=execution,
    )

    cpp = f"""\
    #include <cublasdx.hpp>
    using namespace cublasdx;

    { MM_str }

    using SuggestedLD = suggested_leading_dimension_of_t<MM, { code_type.cc.major * 100 + code_type.cc.minor * 10 }>;
    __device__ constexpr unsigned suggested_leading_dimension_x = SuggestedLD::a;
    __device__ constexpr unsigned suggested_leading_dimension_y = SuggestedLD::b;
    __device__ constexpr unsigned suggested_leading_dimension_z = SuggestedLD::c;
    """

    return {"cpp": cpp}
