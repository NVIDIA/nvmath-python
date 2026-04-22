# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module defines a CUDA emitter for the universal sparse tensor (UST).
"""

__all__ = []

from enum import IntEnum
from io import StringIO

from ._semiring import prolog_epilog_decl, semiring_ops_decl
from ._utils import COMPLEX_PRECISION, type_str
from .tensor_format import Add, Dimension, Divide, LevelExpr, LevelFormat, Modulo, Subtract


class _EmitAction(IntEnum):
    APPLY = 0  # apply value
    DOT = 1  # c = A(i) b(i)
    VM = 2  # c(j) = A(i) B(i,j)
    VM_T = 3  # c(j) = A(i) B(i,j)^T
    MV = 4  # c(i) = A(i,j) b(j)
    BMM = 5  # C(i,k) = A(i,j) B(j,k) or C(b0.., i,k) = A(b0.., i,j) B(j,k)
    BMM_T = 6  # C(i,k) = A(i,j) B(j,k)^T or C(b0.., i,k) = A(b0.., i,j) B(j,k)^T
    BMBM = 7  # C(b0.., i,k) = A(b0.., i,j) B(b0.., j,k)
    BMBM_T = 8  # C(b0.., i,k) = A(b0.., i,j) B(b0.., j,k)^T  (last two dimensions)


def _transposeB(action):
    if action == _EmitAction.DOT:
        return _EmitAction.DOT
    if action == _EmitAction.VM:
        return _EmitAction.VM_T
    if action == _EmitAction.VM_T:
        return _EmitAction.VM
    if action == _EmitAction.BMM:
        return _EmitAction.BMM_T
    if action == _EmitAction.BMM_T:
        return _EmitAction.BMM
    if action == _EmitAction.BMBM:
        return _EmitAction.BMBM_T
    if action == _EmitAction.BMBM_T:
        return _EmitAction.BMBM
    raise ValueError(f"Unexpected transposition of {action}")


def _find_range(litems, expr1, expr2):
    for rl, (k, v) in enumerate(litems):
        if v == LevelFormat.RANGE and k in (expr1, expr2):
            return rl, k
    raise ValueError(f"Cannot find {expr1} or {expr2} in levels")


def _emit_lvl2dim(tensor, indent, stream):
    """Private method to translate levels to dimensions."""
    litems = tensor.tensor_format.levels.items()
    for lvl, (k, fmt) in enumerate(litems):
        if isinstance(fmt, tuple):
            fmt, _ = fmt
        if isinstance(k, Dimension):
            di = tensor.tensor_format.dimensions.index(k)
            print(f"{indent}d{di} = l{lvl};", file=stream)
        elif isinstance(k.operator, Add):
            rl, re = _find_range(litems, k.expression1, k.expression2)
            if re == k.expression2:
                i = tensor.tensor_format.dimensions.index(k.expression1)
                print(f"{indent}d{i} = l{lvl} - l{rl};", file=stream)
            else:
                assert re == k.expression1
                i = tensor.tensor_format.dimensions.index(k.expression2)
                print(f"{indent}d{i} = l{lvl} - l{rl};", file=stream)
        elif isinstance(k.operator, Subtract):
            rl, re = _find_range(litems, k.expression1, k.expression2)
            if re == k.expression2:
                i = tensor.tensor_format.dimensions.index(k.expression1)
                print(f"{indent}d{i} = l{rl} + l{lvl};", file=stream)
            else:
                assert re == k.expression1
                i = tensor.tensor_format.dimensions.index(k.expression2)
                print(f"{indent}d{i} = l{rl} - l{lvl};", file=stream)
        elif isinstance(k.operator, Divide):
            di = tensor.tensor_format.dimensions.index(k.expression1)
            print(f"{indent}d{di} = l{lvl} * {k.expression2};", file=stream)
        elif isinstance(k.operator, Modulo):
            di = tensor.tensor_format.dimensions.index(k.expression1)
            print(f"{indent}d{di} += l{lvl};", file=stream)  # seen second
        else:
            raise AssertionError(f"Unsupported: {k}")


def _emit_types(tensor, ctp, stream):
    """Private method to emit type macros."""
    vtp = type_str(tensor.dtype)
    itp = type_str(tensor.index_type)
    atp = vtp

    if tensor.dtype in ["float8_e4m3fn", "float8_e5m2"]:
        print("#include <cuda_fp8.h>\n", file=stream)
        atp = ctp  # TODO: remove, but to make it compile atomic
    elif tensor.dtype == "float16":
        print("#include <cuda_fp16.h>\n", file=stream)
    elif tensor.dtype == "bfloat16":
        print("#include <cuda_bf16.h>\n", file=stream)
    elif tensor.dtype in ["complex64", "complex128"]:
        print("#include <cuda/std/complex>\n", file=stream)
        atp = f"{COMPLEX_PRECISION[tensor.dtype]}"

    print(f"using VAL = {vtp};", file=stream)
    print(f"using POS = {itp};", file=stream)
    print(f"using CRD = {itp};", file=stream)
    if ctp is not None:
        print(f"using ATP = {atp};", file=stream)
        print(f"using CTP = {ctp};", file=stream)


def _emit_parameters(tensor, A, kind, stream):
    """Private method to emit parameters."""
    for lvl, (_, fmt) in enumerate(tensor.tensor_format.levels.items()):
        if isinstance(fmt, tuple):
            fmt, _ = fmt
        # Handle level format.
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.BATCH or fmt == LevelFormat.RANGE:
            pass
        elif fmt == LevelFormat.COMPRESSED or fmt == LevelFormat.DELTA:
            print(f"  const POS* __restrict__ {A}pos{lvl},", file=stream)
            print(f"  const CRD* __restrict__ {A}crd{lvl},", file=stream)
        elif fmt == LevelFormat.SINGLETON:
            print(f"  const CRD* __restrict__ {A}crd{lvl},", file=stream)
        else:
            raise AssertionError(f"Unsupported: {fmt}")
    print(f"  VAL* __restrict__ {A}values", file=stream, end="\n" if kind == 0 else ",\n")

    # Optionally append dimension or level sizes.
    if kind == 1:
        print(" ", ", ".join([f"unsigned long {A}d{i}" for i in range(tensor.num_dimensions)]), end=",\n", file=stream)
        print(" ", ", ".join([f"unsigned long {A}l{i}" for i in range(tensor.num_levels)]), end=",\n", file=stream)


def _emit_header(stream):
    """Private method to emit header with thread/block logic."""
    print(") {", file=stream)
    print("  const unsigned long tidx = threadIdx.x;", file=stream)
    print("  const unsigned long bidx = blockIdx.x;", file=stream)
    print("  const unsigned long tid = bidx * blockDim.x + tidx;", file=stream)


def _emit_decls(tensor, stream):
    if not tensor.tensor_format.is_identity:
        dims = ", ".join(f"d{d}" for d in range(tensor.num_dimensions))
        print(f"  CRD {dims};", file=stream)
    lvls = ", ".join(f"l{level}" for level in range(tensor.num_levels))
    print(f"  CRD {lvls};", file=stream)


def _emit_load_a(aidx, stream, end=", "):
    print(f"prolog_a(static_cast<CTP>(Avalues[{aidx}]))", end=end, file=stream)


def _emit_load_b(bidx, stream, end=");\n"):
    print(f"prolog_b(static_cast<CTP>(Bvalues[{bidx}]))", end=end, file=stream)


def _emit_load_c(cidx, indent, stream):
    print(f"{indent}CTP acc = prolog_c(static_cast<CTP>(Cvalues[{cidx}]));", file=stream)


def _emit_store_c(cidx, indent, stream):
    print(f"{indent}Cvalues[{cidx}] = static_cast<VAL>(epilog(acc));", file=stream)


def _emit_atomic(dtype, indent, cidx, stream):
    if "complex" in dtype:
        print(f"{indent}  VAL* offset = Cvalues + {cidx};", file=stream)
        print(f"{indent}  ATP* address = reinterpret_cast<ATP *>(offset);", file=stream)
        print(f"{indent}  atomic_add(address + 0, acc.real());", file=stream)
        print(f"{indent}  atomic_add(address + 1, acc.imag());", file=stream)
    else:
        # ATP == VAL
        print(f"{indent}  atomic_add(Cvalues + {cidx}, static_cast<VAL>(acc));", file=stream)


def _emit_traversal(tensor, action, transpose_A, stream):
    # Open the traversal.
    P = None
    lasta = None
    batch = 0
    no_atomic = False
    for lvl, (k, fmt) in enumerate(tensor.tensor_format.levels.items()):
        if isinstance(fmt, tuple):
            fmt, _ = fmt
        lvlsz = tensor.levels[lvl] if action == _EmitAction.APPLY else f"Al{lvl}"
        indent = "  " * (lvl + 1)
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.BATCH:
            if lvl == 0:
                # Dense outermost driver loop.
                print(f"{indent}const CRD p0 = tid;", file=stream)
                print(f"{indent}if (p0 >= {lvlsz})", file=stream)
                print(f"{indent}  return;", file=stream)
                print(f"{indent}l0 = p0;", file=stream)
                if not transpose_A and action == _EmitAction.MV and tensor.tensor_format.is_identity:
                    _emit_load_c("l0", indent, stream)
                    no_atomic = True
                print(f"{indent}{{", file=stream)
            else:
                # Dense inner traversal loop.
                print(f"{indent}POS p{lvl} = p{P} * {lvlsz};", file=stream)
                print(
                    f"{indent}for (CRD i{lvl} = 0; i{lvl} < {lvlsz}; i{lvl}++, p{lvl}++) {{",
                    file=stream,
                )
                print(f"{indent}  l{lvl} = i{lvl};", file=stream)
            P = lvl
            if fmt == LevelFormat.BATCH:
                batch += 1
        elif fmt == LevelFormat.COMPRESSED:
            if lvl == 0:
                # Compressed outermost driver loop.
                print(f"{indent}const POS p0 = tid;", file=stream)
                print(f"{indent}if (p0 >= Apos0[1])", file=stream)
                print(f"{indent}  return;", file=stream)
                print(f"{indent}l0 = Acrd0[p0];", file=stream)
                print(f"{indent}{{", file=stream)
            else:
                # Compressed inner traversal loop.
                if batch > 0:
                    # This only happens for prior BATCH. Correct the higher-dimensions in
                    # both pos and crd buffers and adjust position by #nnz per batch.
                    cidx = f"p{batch - 1} * (Al{batch} + 1) + l{P}"
                    hidx = f"p{batch - 1} * (Al{batch} + 1) + Al{batch}"
                    print(f"{indent}const POS lo{lvl} = Apos{lvl}[{cidx}];", file=stream)
                    print(f"{indent}const POS hi{lvl} = Apos{lvl}[{cidx} + 1];", file=stream)
                    print(f"{indent}const POS no{lvl} = Apos{lvl}[{hidx}];", file=stream)
                    print(
                        f"{indent}for (POS i{lvl} = lo{lvl}, p{lvl} = p{lvl - 2} * no{lvl} "
                        f"+ i{lvl}; i{lvl} < hi{lvl}; i{lvl}++, p{lvl}++) {{",
                        file=stream,
                    )
                    print(f"{indent}  l{lvl} = Acrd{lvl}[p{lvl}];", file=stream)
                    batch = 0  # consumed
                else:
                    print(f"{indent}const POS lo{lvl} = Apos{lvl}[p{P}];", file=stream)
                    print(f"{indent}const POS hi{lvl} = Apos{lvl}[p{P} + 1];", file=stream)
                    print(f"{indent}for (POS p{lvl} = lo{lvl}; p{lvl} < hi{lvl}; p{lvl}++) {{", file=stream)
                    print(f"{indent}  l{lvl} = Acrd{lvl}[p{lvl}];", file=stream)
            P = lvl
            if isinstance(k, LevelExpr) and isinstance(k.operator, (Add, Subtract)):
                assert lasta is None  # no add/sub nesting
                lasta = lvl, k  # record last add/sub
        elif fmt == LevelFormat.SINGLETON:
            assert lvl > 0  # no driver loop
            print(f"{indent}l{lvl} = Acrd{lvl}[p{P}];", file=stream)
            print(f"{indent}{{", file=stream)
        elif fmt == LevelFormat.RANGE:
            assert lvl > 0  # no driver loop
            assert lasta is not None
            la, add = lasta
            isI = k == add.expression2
            di = tensor.tensor_format.dimensions.index(k)
            dj = tensor.tensor_format.dimensions.index(add.expression1 if isI else add.expression2)
            szi = tensor.extents[di] if action == _EmitAction.APPLY else f"Ad{di}"
            szj = tensor.extents[dj] if action == _EmitAction.APPLY else f"Ad{dj}"
            if isinstance(add.operator, Add):
                print(f"{indent}const CRD of{lvl} = l{la};", file=stream)
                print(f"{indent}const CRD lo{lvl} = llmax(0, of{lvl} - {szj} + 1);", file=stream)
                print(f"{indent}const CRD hi{lvl} = llmin({szi}, of{lvl} + 1);", file=stream)
            else:
                sign = "-" if isI else ""
                print(f"{indent}const CRD of{lvl} = {sign}l{la};", file=stream)
                print(f"{indent}const CRD lo{lvl} = llmax(0, of{lvl});", file=stream)
                print(f"{indent}const CRD hi{lvl} = llmin({szi}, {szj} + of{lvl});", file=stream)
            print(
                f"{indent}POS p{lvl} = p{P} * {szi} + lo{lvl};",
                file=stream,
            )
            print(
                f"{indent}for (CRD i{lvl} = lo{lvl}; i{lvl} < hi{lvl}; i{lvl}++, p{lvl}++) {{",
                file=stream,
            )
            print(f"{indent}  l{lvl} = i{lvl};", file=stream)
            P = lvl
        elif fmt == LevelFormat.DELTA:
            assert lvl > 0  # no driver loop
            # Delta compressed inner traversal loop.
            print(f"{indent}const POS lo{lvl} = Apos{lvl}[p{P}];", file=stream)
            print(f"{indent}const POS hi{lvl} = Apos{lvl}[p{P} + 1];", file=stream)
            print(
                f"{indent}for (POS p{lvl} = lo{lvl}, at{lvl} = 0; p{lvl} < hi{lvl}; p{lvl}++, at{lvl}++) {{",
                file=stream,
            )
            print(f"{indent}  at{lvl} += Acrd{lvl}[p{lvl}];", file=stream)
            print(f"{indent}  l{lvl} = at{lvl};", file=stream)
            P = lvl
        else:
            raise AssertionError(f"Unsupported: {fmt}")
        lvl += 1
    # Translation prior to traversal.
    indent = "  " * (tensor.num_levels + 1)
    if tensor.tensor_format.is_identity:
        dvars = [f"l{level}" for level in range(tensor.num_levels)]
    else:
        _emit_lvl2dim(tensor, indent, stream)
        dvars = [f"d{d}" for d in range(tensor.num_dimensions)]
    # Transposition of A (last two dimensions).
    if transpose_A:
        dvars[-2], dvars[-1] = dvars[-1], dvars[-2]
    # Perform the traversal.
    if action == _EmitAction.APPLY:  # apply
        idxs = ", ".join(d for d in dvars)
        print(f"{indent}Avalues[p{P}] = apply(Avalues[p{P}], {idxs});", file=stream)
    elif action == _EmitAction.DOT:  # c = A(i) b(i)
        print(f"{indent}{{", file=stream)
        print(f"{indent}  const CTP acc = mul(", end="", file=stream)
        _emit_load_a(f"p{P}", stream)
        _emit_load_b(f"{dvars[0]}", stream)
        _emit_atomic(tensor.dtype, indent, "0", stream)
        print(f"{indent}}}", file=stream)
    elif action == _EmitAction.VM or action == _EmitAction.VM_T:  # c(k) = A(j) B(j,k)
        print(f"{indent}for (CRD k = 0; k < Bd1; k++) {{", file=stream)
        print(f"{indent}  const CTP acc = mul(", end="", file=stream)
        _emit_load_a(f"p{P}", stream)
        bidx = f"{dvars[0]} * Bd1 + k" if action == _EmitAction.VM else f"{dvars[0]} + Bd0 * k"
        _emit_load_b(f"{bidx}", stream)
        _emit_atomic(tensor.dtype, indent, "k", stream)
        print(f"{indent}}}", file=stream)
    elif action == _EmitAction.MV:  # c(i) = A(i,j) b(j)
        print(f"{indent}{{", file=stream)
        if no_atomic:
            print(f"{indent}  acc = add(acc, mul(", end="", file=stream)
            _emit_load_a(f"p{P}", stream)
            _emit_load_b(f"{dvars[1]}", stream, end="));\n")
        else:
            print(f"{indent}  const CTP acc = mul(", end="", file=stream)
            _emit_load_a(f"p{P}", stream)
            _emit_load_b(f"{dvars[1]}", stream)
            _emit_atomic(tensor.dtype, indent, f"{dvars[0]}", stream)
        print(f"{indent}}}", file=stream)
    elif action == _EmitAction.BMM or action == _EmitAction.BMM_T:  # C(b0.., i,k) = A(b0.., i,j) B(j,k)
        b = tensor.num_dimensions - 1
        print(f"{indent}for (CRD k = 0; k < Bd1; k++) {{", file=stream)
        print(f"{indent}  const CTP acc = mul(", end="", file=stream)
        _emit_load_a(f"p{P}", stream)
        bidx = f"{dvars[b]} * Bd1 + k" if action == _EmitAction.BMM else f"{dvars[b]} + Bd0 * k"
        _emit_load_b(f"{bidx}", stream)
        cidx = f"{dvars[0]}"
        for d in range(1, b):
            cidx = f"({cidx} * Ad{d} + {dvars[d]})"
        _emit_atomic(tensor.dtype, indent, f"{cidx} * Bd1 + k", stream)
        print(f"{indent}}}", file=stream)
    elif action == _EmitAction.BMBM or action == _EmitAction.BMBM_T:  # C(b0.., i,k) = A(b0.., i,j) B(b0.., j,k)
        b = tensor.num_dimensions - 1
        print(f"{indent}for (CRD k = 0; k < Bd{b}; k++) {{", file=stream)
        print(f"{indent}  const CTP acc = mul(", end="", file=stream)
        _emit_load_a(f"p{P}", stream)
        bidx = f"{dvars[0]}"
        for d in range(1, b - 1):
            bidx = f"({bidx} * Ad{d} + {dvars[d]})"
        bidx = (
            f"(({bidx} * Bd{b - 1}) + {dvars[b]}) * Bd{b} + k"
            if action == _EmitAction.BMBM
            else f"(({bidx} * Bd{b - 1}) + k) * Bd{b} + {dvars[b]}"
        )
        _emit_load_b(f"{bidx}", stream)
        cidx = f"{dvars[0]}"
        for d in range(1, b):
            cidx = f"({cidx} * Ad{d} + {dvars[d]})"
        _emit_atomic(tensor.dtype, indent, f"{cidx} * Bd{b} + k", stream)
        print(f"{indent}}}", file=stream)
    else:
        raise TypeError(f"Unsupported: {action}")
    # Close the traversal.
    for level in range(tensor.num_levels, 0, -1):
        indent = "  " * level
        print(f"{indent}}}", file=stream)
    if no_atomic:
        _emit_store_c("l0", indent, stream)


def emit_apply(tensor, with_indices):
    """
    Emits CUDA code for an apply() operation. Note that since the kernel
    is very specific to one UST instance, the kernel can freely specialize
    on metadata like dimension or level sizes.
    """
    stream = StringIO()
    _emit_types(tensor, None, stream)
    dim = tensor.num_dimensions if with_indices else 0
    print(f'\nextern "C" __device__ VAL apply(VAL{", CRD" * dim});\n', file=stream)
    print('extern "C" __global__ void apply_kernel(', file=stream)
    if with_indices:
        _emit_parameters(tensor, "A", 0, stream)
        _emit_header(stream)
        _emit_decls(tensor, stream)
        _emit_traversal(tensor, _EmitAction.APPLY, False, stream)
    else:
        # Fast path for values only.
        print("  VAL* __restrict__ Avalues,", file=stream)
        print("  POS Anse", file=stream)
        _emit_header(stream)
        print("  if (tid >= Anse)", file=stream)
        print("    return;", file=stream)
        print("  Avalues[tid] = apply(Avalues[tid]);", file=stream)
    print("}", file=stream)
    return stream.getvalue()


# TODO: proof-of-concept sparse emitter that should be generalized; currently
#       it only accepts very simple forms where only operand A is sparse;
#       also, currently it drops the prolog_c/epilog when atomic updates are needed
def _actions(tensor_A, tensor_B, tensor_C):
    """Returns (akind, bkind, action) encoding for supported matmul combinations."""

    if tensor_B.tensor_format.name not in [
        "DenseVector",
        "DensedRight",
        "DensedLeft",
        "Dense3D-0-1-2",
        "Dense4D-0-1-2-3",
    ] or tensor_C.tensor_format.name not in ["Scalar", "DenseVector", "DensedRight", "Dense3D-0-1-2", "Dense4D-0-1-2-3"]:
        raise NotImplementedError(
            f"The code generator doesn't currently support the \
{tensor_B.tensor_format.name} or {tensor_C.tensor_format.name} formats."
        )

    if tensor_A.num_dimensions == 1:
        # A is a vector.
        if tensor_B.num_dimensions == 1 and tensor_C.num_dimensions == 0:
            return 1, 1, _EmitAction.DOT  # c = A(i) b(i)
        if tensor_B.num_dimensions == 2 and tensor_C.num_dimensions == 1:
            if tensor_B.tensor_format.name != "DensedLeft":
                return 1, 1, _EmitAction.VM  # c(j) = A(i) B(i,j)
            else:
                return 1, 1, _EmitAction.VM_T  # c(j) = A(i) B(i,j)^T
    elif tensor_A.num_dimensions == 2:
        # A is a matrix.
        if tensor_B.num_dimensions == 1 and tensor_C.num_dimensions == 1:
            return 1, -1, _EmitAction.MV  # c(i) = A(i,j) b(j)
        elif tensor_B.num_dimensions == 2 and tensor_C.num_dimensions == 2:
            if tensor_B.tensor_format.name != "DensedLeft":
                return 1, 1, _EmitAction.BMM  # C(i,k) = A(i,j) B(j,k)
            else:
                return 1, 1, _EmitAction.BMM_T  # C(i,k) = A(i,j) B(j,k)^T
    elif tensor_A.num_dimensions > 2:
        # A is a tensor.
        if tensor_B.num_dimensions == 2 and tensor_C.num_dimensions == tensor_A.num_dimensions:
            if tensor_B.tensor_format.name != "DensedLeft":
                return 1, 1, _EmitAction.BMM  # C(b0.., i,k) = A(b0.., i,j) B(j,k)
            else:
                return 1, 1, _EmitAction.BMM_T  # C(b0.., i,k) = A(b0.., i,j) B(j,k)^T
        elif tensor_B.num_dimensions == tensor_A.num_dimensions and tensor_C.num_dimensions == tensor_A.num_dimensions:
            return 1, 1, _EmitAction.BMBM  # C(b0.., i,k) = A(b0.., i,j) B(b0.., j,k)
    raise TypeError("Unsupported matmul() in sparse emitter")


def emit_matmul(tensor_A, tensor_B, tensor_C, ctp, transpose_A=False, transpose_B=False):
    """
    Emits CUDA code for a matmul(A, B, C) operation. Note that since the kernel
    is cached at a generic UST and type level only, the kernel should not specialize
    on metadata like dimension or level sizes, but use parameter values instead.
    """
    akind, bkind, action = _actions(tensor_A, tensor_B, tensor_C)
    stream = StringIO()
    _emit_types(tensor_A, ctp, stream)
    print(f"{semiring_ops_decl}{prolog_epilog_decl}", file=stream)
    print('extern "C" __global__ void matmul(', file=stream)
    _emit_parameters(tensor_A, "A", akind, stream)
    _emit_parameters(tensor_B, "B", bkind, stream)
    _emit_parameters(tensor_C, "C", 0, stream)
    _emit_header(stream)
    _emit_decls(tensor_A, stream)
    if transpose_B:
        action = _transposeB(action)
    _emit_traversal(tensor_A, action, transpose_A, stream)
    print("}", file=stream)
    return stream.getvalue()


def _populate_sizes(params, tensor):
    for i in range(tensor.num_dimensions):
        params.append(tensor.extents[i])
    for i in range(tensor.num_levels):
        params.append(tensor.levels[i])
    return params


# TODO: improve the parallelization scheme dramatically
def _populate_parameter(params, tensor, kind):
    """Private method to populate parameters for a single UST tensor."""
    size = None
    for lvl, (_, fmt) in enumerate(tensor.tensor_format.levels.items()):
        if isinstance(fmt, tuple):
            fmt, _ = fmt
        # Handle level format.
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.BATCH or fmt == LevelFormat.RANGE:
            if lvl == 0:
                size = tensor.levels[0]
            else:
                size = tensor.extents[0] if size < tensor.extents[0] else size
        elif fmt == LevelFormat.COMPRESSED or fmt == LevelFormat.DELTA:
            params.append(tensor.pos(lvl).data_ptr)
            params.append(tensor.crd(lvl).data_ptr)
            if lvl == 0:
                size = tensor.crd(0).size
        elif fmt == LevelFormat.SINGLETON:
            params.append(tensor.crd(lvl).data_ptr)
        else:
            raise TypeError(f"Unsupported: {fmt}")
    params.append(tensor.val.data_ptr)

    # Optionally append dimension and level sizes.
    if kind == 1:
        params = _populate_sizes(params, tensor)

    return (params, size)


def populate_apply_parameters(tensor, with_indices=True):
    """
    Populates the parameters for an apply() operation (with our without indices).
    The API follows the UST parameter passing conventions for the tensor when
    indices are needed. Otherwise it directly passes the values buffer.

    Returns (parameter list, kernel launch size)
    """
    if with_indices:
        return _populate_parameter([], tensor, 0)
    return ([tensor.val.data_ptr, tensor.nse], tensor.nse)


def populate_matmul_parameters(tensor_A, tensor_B, tensor_C, dense_bc=False):
    """
    Populates the parameters for a matmul(A, B, C) operation.

    The API follows the UST parameter passing conventions for A, B, C,
    each possibly followed by the dimension and level sizes.

    Returns (parameter list, kernel launch size)
    """
    akind, bkind, _ = _actions(tensor_A, tensor_B, tensor_C)
    params = []
    params, n = _populate_parameter(params, tensor_A, akind)
    if dense_bc:
        # Fast path for dense B and C.
        params.append(tensor_B._val.data_ptr)
        if bkind == 1:
            params = _populate_sizes(params, tensor_B)
        params.append(tensor_C._val.data_ptr)
    else:
        params, _ = _populate_parameter(params, tensor_B, bkind)
        params, _ = _populate_parameter(params, tensor_C, 0)
    assert n is not None
    return (params, n)
