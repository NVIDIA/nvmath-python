# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module overrides PyTorch tensors with the UST (for easy "injection").
"""

__all__ = ["TorchUST", "prune_model", "reformat_model"]


import numbers
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.utils.prune as prune

from nvmath.sparse.generic import Matmul
from nvmath.sparse.ust._utils import Cache
from nvmath.sparse.ust.tensor import Tensor

_MATMUL_CACHE = Cache()


def _matmul(A, B, C_shape, transpose_a=False):
    a = A.ust
    b = Tensor.from_package(B)

    vtp = a.dtype
    itp = a.index_type
    format_a = a.tensor_format.name
    format_b = b.tensor_format.name
    sizes = (tuple(a.extents), tuple(b.extents), C_shape)

    # Lookup matmul. Also recycles the result tensor allocation.
    key = (vtp, itp, format_a, format_b, transpose_a, sizes, A.stream, A.device)
    mm = _MATMUL_CACHE.get(key)
    if mm is None:
        t = torch.zeros(C_shape, dtype=A.dtype, device=A.device)
        c = Tensor.from_package(t)
        # Or: options = {"codegen": True}
        options = None
        if transpose_a:
            matrix_qualifiers_dtype = np.dtype([("is_transpose", "<i4"), ("is_conjugate", "<i4")])
            qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
            qualifiers[0]["is_transpose"] = True
        else:
            qualifiers = None
        mm = Matmul(a, b, c, beta=1.0, qualifiers=qualifiers, options=options)
        mm.plan()
        _MATMUL_CACHE[key] = (mm, t, c)
    else:
        mm, t, c = mm
        t.zero_()
        mm.reset_operands_unchecked(a=a, b=b, c=c)
    mm.execute()
    mm.release_operands()

    return t  # fast c.to_package()


def _ust_detach(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    return self.__class__(
        shape=self.shape,
        device=self.device,
        dtype=self.dtype,
        layout=self.layout,
        requires_grad=False,
        ust=self.ust,
        transposed=self.transposed,
    )


def _ust_zeros_like(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    return torch.zeros(self.shape, dtype=self.dtype, device=self.device)


def _ust_ones_like(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    return torch.ones(self.shape, dtype=self.dtype, device=self.device)


def _ust_empty_like(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    return torch.empty(self.shape, dtype=self.dtype, device=self.device)


def _ust_to_copy(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    return self


def _ust_t(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, TorchUST)
    lshape = len(self.shape)
    # TODO: self.requires_grad?
    if lshape == 1:
        return self.__class__(
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            layout=self.layout,
            requires_grad=False,
            ust=self.ust,
            transposed=False,  # no impact on vector
        )
    elif lshape == 2:
        return self.__class__(
            shape=(self.shape[1], self.shape[0]),  # type: ignore
            device=self.device,
            dtype=self.dtype,
            layout=self.layout,
            requires_grad=False,
            ust=self.ust,
            transposed=not self.transposed,  # simply flip
        )
    raise NotImplementedError("`TorchUST` transpose: Unexpected operand")


def _ust_permute(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 2
    self, perm = args
    assert isinstance(self, TorchUST)
    if len(perm) == 1 or (len(perm) == 2 and perm[0] == 1 and perm[1] == 0):
        return _ust_t(func, types, args[:-1], kwargs)
    raise NotImplementedError("`TorchUST` transpose: Unexpected operand")


def _ust_view(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 2
    self, perm = args
    raise NotImplementedError("`TorchUST` view: Unexpected operand")


def _ust_dot_impl(A, B):
    assert A.ndim == 1 and B.ndim == 1
    if isinstance(A, TorchUST):
        if isinstance(B, TorchUST):
            raise NotImplementedError("`TorchUST` dot: Only one operand can be sparse")
        # DOT a x b direct, can ignore A.transposed.
        return _matmul(A, B, ())
    elif isinstance(B, TorchUST):
        # DOT a x b as b x a, can ignore B.transposed.
        return _matmul(B, A, ())
    raise NotImplementedError("`TorchUST` dot: Unexpected operands")


def _ust_mv_impl(A, B):
    assert A.ndim == 2 and B.ndim == 1
    if isinstance(A, TorchUST):
        if isinstance(B, TorchUST):
            raise NotImplementedError("`TorchUST` mv: Only one operand can be sparse")
        # MV A* x b direct.
        return _matmul(A, B, (A.shape[0],), transpose_a=A.transposed)
    elif isinstance(B, TorchUST):
        # MV A x b as b x A^T, can ignore sparse vector B.transposed.
        return _matmul(B, A.t(), (A.shape[0],))
    raise NotImplementedError("`TorchUST` mv: Unexpected operands")


def _ust_mm_impl(A, B):
    if isinstance(A, TorchUST):
        if isinstance(B, TorchUST):
            raise NotImplementedError("`TorchUST` mm: Only one operand can be sparse")
        # MM A* x B direct.
        return _matmul(A, B, (*A.shape[:-1], B.shape[-1]), transpose_a=A.transposed)
    elif isinstance(B, TorchUST):
        assert A.ndim == 2 and B.ndim == 2
        # MM A x B* as (B** x A^T)^T.
        return _matmul(B, A.t(), (B.shape[1], A.shape[0]), transpose_a=not B.transposed).t()
    raise NotImplementedError("`TorchUST` mm: Unexpected operands")


def _ust_dot(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    A, B = args
    C = _ust_dot_impl(A, B)
    # print("ust_dot(", A.shape, ",", B.shape, ") ->", C.shape)
    return C


def _ust_mv(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) in [2, 3]
    A, B = args[-2:]
    C = _ust_mv_impl(A, B)
    # print("ust_mv(", A.shape, ",", B.shape, ") ->", C.shape)
    return C if len(args) == 2 else C + args[0]


def _ust_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) in [2, 3]
    A, B = args[-2:]
    # if A.shape[0] == 1:
    #    C = _ust_mv_impl(B.t(), A.squeeze(dim=0)).unsqueeze(dim=0)
    #    # print("ust_mm(", A.shape, ",", B.shape, ") ->", C.shape, "as vec")
    # else:
    C = _ust_mm_impl(A, B)
    # print("ust_mm(", A.shape, ",", B.shape, ") ->", C.shape)
    return C if len(args) == 2 else C + args[0]


def _ust_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) in [2, 3]
    A, B = args[:2]  # bias at end
    if A.ndim == 1:
        # Recognized matvec (e.g. single token inference).
        C = _ust_mv_impl(B, A)
        # print("ust_linear_1(", A.shape, ",", B.shape, ") ->", C.shape)
    else:
        # Cast into matmul (works for matvec as well).
        shape = A.shape
        A_2d = A.view(-1, shape[-1])
        C = _ust_mm_impl(A_2d, B.t()).view(*shape[:-1], -1)
        # print("ust_linear(", shape, "->", A_2d.shape, ",", B.shape, ") ->", C.shape)
    return C if len(args) == 2 else C + args[2]


def _ust_mul_inplace(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    A, B = args
    # This only works for broadcasting scalar B because otherwise there
    # is no direct match between values array (nse) and shape of B (all).
    assert isinstance(A, TorchUST)
    if not isinstance(B, numbers.Number):
        raise NotImplementedError("`TorchUST` mul: Unexpected operands")
    # print("ust_mul_inplace(", A.shape, ",", B, ")")
    v = A.ust.val.tensor
    v *= B  # good performance
    return A


def _ust_addcdiv_inplace(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 3
    A, B, C = args
    assert isinstance(A, TorchUST)
    assert not isinstance(B, TorchUST)
    assert not isinstance(C, TorchUST)
    # print("ust_addcdiv_inplace(", A.shape, ",", B.shape, ",", C.shape, ")", kwargs)
    val = kwargs["value"]
    if val is not None:
        raise NotImplementedError("`TorchUST` no inplace addcdiv")
    return A


def _ust_multi_head_attention(func, types, args=(), kwargs=None) -> torch.Tensor:
    raise NotImplementedError("`TorchUST` no multi-head attention (try running in training mode)")


class TorchUST(torch.Tensor):
    """
    This class wraps the universal sparse tensor as a :class:`torch.Tensor`.

    The purpose of this class is to inject the UST into PyTorch models with minimal
    code changes. Objects of the ``TorchUST`` class behave like PyTorch tensors, but
    transparently use an UST implementation for the actual operations. A set of
    operations commonly used in models is supported (otherwise a "can't perform"
    message is prompted; please let us know of such cases, so we can extend the
    supported set accordingly).

    Examples:
        >>> import torch
        >>> from nvmath.sparse.ust.interfaces.torch_interface import TorchUST

        Create two torch vectors.

        >>> x = (1.0 + torch.arange(32)).cuda()
        >>> y = (2.0 + torch.arange(32)).cuda()

        Now perform a dot product operation.

        >>> z = torch.dot(x, y)

        Convert the first operand to UST.

        >>> x = TorchUST.from_torch(x)

        Now perform the same dot production operation. It transparently
        uses the UST implementation!

        >>> z = torch.dot(x, y)
    """

    SPARSE_DISPATCH: dict[Callable, Callable]

    ust: Tensor
    transposed: bool
    stream: int

    __slots__ = ["ust", "transposed"]

    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        layout: torch.layout,
        requires_grad: bool,
        ust: Tensor,
        transposed: bool,
    ):
        # print("new", cls, "(", shape, ",", dtype, layout, ", transp =", transposed, ")")
        cls._load_dispatch_table()
        torch._dynamo.allow_in_graph(cls)
        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=device,
            dtype=dtype,
            layout=torch.strided,  # cheat
            requires_grad=requires_grad,
        )
        tensor.ust = ust
        tensor.transposed = transposed
        tensor.stream = torch.cuda.current_stream().cuda_stream
        return tensor

    @classmethod
    def from_torch(cls, original_tensor: torch.Tensor) -> "TorchUST":
        """Constructs a ``TorchUST`` from the given ``torch.Tensor``."""
        ust = Tensor.from_package(original_tensor)
        return cls(
            original_tensor.shape,
            device=original_tensor.device,
            dtype=original_tensor.dtype,
            layout=original_tensor.layout,
            requires_grad=original_tensor.requires_grad,
            ust=ust,
            transposed=False,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:  # type: ignore[override]
        if func._overloadpacket not in cls.SPARSE_DISPATCH:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__} aka {func._overloadpacket})"
            )
        return cls.SPARSE_DISPATCH[func._overloadpacket](func, types, args, kwargs)

    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None) -> None:
        """
        Loads the op overload sparse dispatch table for the current class.
        """
        if getattr(cls, "SPARSE_DISPATCH", None) is None:
            cls.SPARSE_DISPATCH = {
                torch.ops.aten.detach: _ust_detach,
                torch.ops.aten.zeros_like: _ust_zeros_like,
                torch.ops.aten.ones_like: _ust_ones_like,
                torch.ops.aten.empty_like: _ust_empty_like,
                torch.ops.aten._to_copy: _ust_to_copy,
                torch.ops.aten.t: _ust_t,  # .t()
                torch.ops.aten.permute: _ust_permute,  # .T
                torch.ops.aten.view: _ust_view,
                torch.ops.aten.dot: _ust_dot,
                torch.ops.aten.mv: _ust_mv,
                torch.ops.aten.addmv: _ust_mv,
                torch.ops.aten.mm: _ust_mm,
                torch.ops.aten.addmm: _ust_mm,
                torch.ops.aten.matmul: _ust_mm,
                torch.ops.aten.linear: _ust_linear,
                torch.ops.aten.bmm: _ust_mm,
                torch.ops.aten.baddbmm: _ust_mm,
                torch.ops.aten.mul_: _ust_mul_inplace,
                torch.ops.aten.addcdiv_: _ust_addcdiv_inplace,
                torch.ops.aten._native_multi_head_attention: _ust_multi_head_attention,
            }
            if custom_dispatch_table is not None:
                cls.SPARSE_DISPATCH.update(custom_dispatch_table)

    def __str__(self):
        return self.ust.__str__()


def prune_model(model, *, local=True, amount=0.5):
    """
    This is a convenience wrapper that uses the framework :mod:`torch.nn.utils.prune`
    to prune all the weights of linear layers in a model either locally (per layer)
    or globally (over all layers) with the given amount.

    Args:
        model: the model to be pruned
        local: local per layer pruning if True, otherwise global over all layers
        amount: amount of weights to be dropped (e.g. 0.50 drops 50%)
    """
    # Local pruning.
    if local:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
        return
    # Global pruning.
    parameters_to_prune = tuple([(module, "weight") for module in model.modules() if isinstance(module, torch.nn.Linear)])
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def reformat_model(model, *, func=None):
    """
    This function potentially converts the linear weights in a model into UST format.
    If ``func`` is None, all weights are unconditionally converted into the UST COO format
    (stress testing the system with exactly the same weights but using sparse operations).
    If a user-defined function ``func=reformat`` is given, this method is applied to every
    weight and replaces the parameter only if the function returns not None.

    Args:
        model: the model to be reformatted
        func: if set, user-defined reformatting function

    Examples:
        >>> import torch
        >>> from nvmath.sparse.ust.interfaces.torch_interface import TorchUST

        Inside the ``reformat`` method, inspect the weight sparsity (note that we
        could even prune here, but it is more common to rely on other pruning
        frameworks like ``torch.nn.utils.prune`` in combination with fine-tuning
        for accuracy).

        If the condition is met, pick a suitable format for ``weight`` and then
        return ``TorchUST.from_torch(weight)``. Otherwise, just return ``None``.

        >>> def reformat(weight):
        ...     nel = weight.numel()
        ...     nnz = torch.count_nonzero(weight)
        ...     sparsity = (1.0 - float(nnz) / float(nel)) * 100.0
        ...     if sparsity >= sparse_threshold:
        ...         # TODO: Pick suitable format for weight
        ...         return TorchUST.from_torch(weight)
        ...     return None

        This approach enables experimenting with novel formats to speedup
        sparsified models during inference by simply calling the method
        ``ust.reformat_model(model, func=reformat)``.
        No source code changes inside the model are required! If used during
        training, always make sure to construct the optimizer (by calling e.g.
        ``torch.optim.Adam(model.parameters(), lr=0.001)``)
        after the reformatting method call, so that the new parameters will
        be involved in the optimizer steps.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Make any pruning permanent prior to reformatting
            # (this is required to get a proper weight matrix).
            if prune.is_pruned(module):
                print(f"Making pruning in {name} permanent prior to reformatting")
                prune.remove(module, name="weight")

            # Apply unconditional reformatting into COO if func is None.
            # Otherwise, apply the user-defined function.
            if func is None:
                U = TorchUST.from_torch(module.weight.data.to_sparse())
                module.weight = torch.nn.Parameter(U)
            else:
                U = func(module.weight.data)
                if U is not None:
                    module.weight = torch.nn.Parameter(U)
