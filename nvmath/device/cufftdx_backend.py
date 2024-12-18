# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .caching import json_hash
from .common import check_contains, check_in, check_not_in, check_code_type
from .common_cpp import generate_type_map, NP_TYPES_TO_CPP_TYPES
from .types import REAL_NP_TYPES


def validate(size, precision, fft_type, execution, direction, ffts_per_block, elements_per_thread, real_fft_options, code_type):
    if size <= 0:
        raise ValueError(f"size must be > 0. Got {size}")
    check_in("precision", precision, REAL_NP_TYPES)
    check_in("fft_type", fft_type, ["c2c", "c2r", "r2c"])
    check_in("execution", execution, ["Block", "Thread"])
    if direction is not None:
        check_in("direction", direction, ["forward", "inverse"])
    if ffts_per_block in (None, "suggested"):
        pass
    else:
        if ffts_per_block <= 0:
            raise ValueError(
                f"ffts_per_block must be None, 'suggested' or a positive integer ; got ffts_per_block = {ffts_per_block}"
            )
    if elements_per_thread in (None, "suggested"):
        pass
    else:
        if elements_per_thread <= 0:
            raise ValueError(
                f"elements_per_thread must be None, 'suggested' or a positive integer ; "
                f"got elements_per_thread = {elements_per_thread}"
            )
    if real_fft_options is None:
        pass
    else:
        check_contains(real_fft_options, "complex_layout")
        check_contains(real_fft_options, "real_mode")
        check_in("real_fft_options['complex_layout']", real_fft_options["complex_layout"], ["natural", "packed", "full"])
        check_in("real_fft_options['real_mode']", real_fft_options["real_mode"], ["normal", "folded"])
    check_code_type(code_type)


def generate_FFT(
    size, precision, fft_type, direction, code_type, execution, ffts_per_block, elements_per_thread, real_fft_options
):
    check_not_in("ffts_per_block", ffts_per_block, ["suggested"])
    check_not_in("elements_per_thread", elements_per_thread, ["suggested"])

    if real_fft_options is not None:
        ll = real_fft_options["complex_layout"]
        mm = real_fft_options["real_mode"]
        real_fft_options_str = f"+ RealFFTOptions<complex_layout::{ll}, real_mode::{mm}>()"
    else:
        real_fft_options_str = ""

    if execution == "Block":
        execution = "+ Block()"
    elif execution == "Thread":
        execution = "+ Thread()"
    else:
        raise ValueError(f"execution should be Block or Thread ; got execution = {execution}")

    if code_type is not None:
        sm = f"+ SM<{ code_type.cc.major * 100 + code_type.cc.minor * 10 }>()"
    else:
        sm = ""

    if ffts_per_block is not None:
        ffts_per_block = f"+ FFTsPerBlock<{ ffts_per_block }>()"
    else:
        ffts_per_block = ""

    if elements_per_thread is not None:
        elements_per_thread = f"+ ElementsPerThread<{ elements_per_thread }>()"
    else:
        elements_per_thread = ""

    cpp = f"""\
    using FFT = decltype(  Size<{ size }>() + Precision<{ NP_TYPES_TO_CPP_TYPES[precision] }>()
                         + Type<fft_type::{ fft_type }>() + Direction<fft_direction::{ direction }>()
                         { sm }
                         { real_fft_options_str }
                         { ffts_per_block }
                         { elements_per_thread }
                         { execution }
                         );
    """

    name = json_hash(
        size=size,
        precision=NP_TYPES_TO_CPP_TYPES[precision],
        fft_type=fft_type,
        direction=direction,
        sm=sm,
        ffts_per_block=ffts_per_block,
        elements_per_thread=elements_per_thread,
        real_fft_options=real_fft_options,
    )

    return cpp, name


def generate_block(size, precision, fft_type, direction, code_type, ffts_per_block, elements_per_thread, real_fft_options):
    FFT, name = generate_FFT(
        size=size,
        precision=precision,
        fft_type=fft_type,
        direction=direction,
        code_type=code_type,
        execution="Block",
        ffts_per_block=ffts_per_block,
        elements_per_thread=elements_per_thread,
        real_fft_options=real_fft_options,
    )

    TYPE_MAP, type_map_name = generate_type_map(name=name)

    thread_api_name = f"libmathdx_function_fft_thread_{name}"
    smem_api_name = f"libmathdx_function_fft_smem_{name}"

    cpp = f"""\
    #include <cufftdx.hpp>
    using namespace cufftdx;

    { TYPE_MAP }

    { FFT }

    __device__ constexpr unsigned  block_dim_x = FFT::block_dim.x;
    __device__ constexpr unsigned  block_dim_y = FFT::block_dim.y;
    __device__ constexpr unsigned  block_dim_z = FFT::block_dim.z;
    __device__ constexpr unsigned  storage_size = FFT::storage_size;
    __device__ constexpr unsigned  shared_memory_size = FFT::shared_memory_size;
    __device__ constexpr unsigned  suggested_ffts_per_block = FFT::suggested_ffts_per_block;
    __device__ constexpr unsigned  stride = FFT::stride;
    __device__ constexpr unsigned  ffts_per_block = FFT::ffts_per_block;
    __device__ constexpr unsigned  elements_per_thread = FFT::elements_per_thread;
    __device__ constexpr unsigned  implicit_type_batching = FFT::implicit_type_batching;
    __device__ constexpr bool      requires_workspace = FFT::requires_workspace;
    __device__ constexpr unsigned  workspace_size = FFT::workspace_size;
    __device__ constexpr unsigned  value_type = {type_map_name}<FFT::value_type>::value;
    __device__ constexpr unsigned  input_type = {type_map_name}<FFT::input_type>::value;
    __device__ constexpr unsigned  output_type = {type_map_name}<FFT::output_type>::value;

    __device__ void { thread_api_name }(FFT::value_type* rmem, FFT::value_type* smem, void* handle) {{
        FFT().execute(rmem, smem);
    }}

    __device__ void { smem_api_name }(FFT::value_type* rmem, FFT::value_type* smem, void* handle) {{
        FFT().execute(smem);
    }}

    """

    return {"cpp": cpp, "names": {"thread": thread_api_name, "smem": smem_api_name}}


def generate_thread(size, precision, fft_type, direction, code_type, real_fft_options):
    FFT, name = generate_FFT(
        size=size,
        precision=precision,
        fft_type=fft_type,
        direction=direction,
        code_type=code_type,
        execution="Thread",
        ffts_per_block=None,
        elements_per_thread=None,
        real_fft_options=real_fft_options,
    )
    thread_api_name = f"libmathdx_function_fft_thread_{name}"

    TYPE_MAP, type_map_name = generate_type_map(name=name)

    cpp = f"""\
    #include <cufftdx.hpp>
    using namespace cufftdx;

    { TYPE_MAP }

    { FFT }

    __device__ constexpr unsigned  storage_size = FFT::storage_size;
    __device__ constexpr unsigned  stride = FFT::stride;
    __device__ constexpr unsigned  elements_per_thread = FFT::elements_per_thread;
    __device__ constexpr unsigned  implicit_type_batching = FFT::implicit_type_batching;
    __device__ constexpr unsigned  value_type = {type_map_name}<FFT::value_type>::value;
    __device__ constexpr unsigned  input_type = {type_map_name}<FFT::input_type>::value;
    __device__ constexpr unsigned  output_type = {type_map_name}<FFT::output_type>::value;

    __device__ void { thread_api_name }(FFT::value_type* rmem) {{
        FFT().execute(rmem);
    }}
    """

    return {"cpp": cpp, "names": {"thread": thread_api_name}}
