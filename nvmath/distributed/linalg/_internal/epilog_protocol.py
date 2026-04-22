# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Internal interface for epilog input handling."""

__all__ = [
    "EPILOG_INPUT_HANDLERS_MAP",
    "EPILOG_OUTPUT_HANDLERS_MAP",
    "BiasHandler",
    "BgradHandler",
    "ReluAuxHandler",
    "GeluAuxHandler",
    "DGeluAuxHandler",
    "EpilogInputHandler",
    "EpilogOutputHandler",
]

import math
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from nvmath.bindings import cublasMp  # type: ignore
from nvmath.internal import typemaps
from nvmath.linalg._internal.utils import axis_order_in_memory, calculate_strides, check_batch_tileable

Epilog = cublasMp.MatmulEpilogue


@runtime_checkable
class EpilogInputHandler(Protocol):
    """
    Protocol for epilog handler input validation and setting the appropriate MM descriptor
    attributes.
    """

    @abstractmethod
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """
        The name of the epilog input that is handled (bias, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, epilog_input):
        """
        Validate the provided epilog input.

        Args:
            epilog_input: The epilog input to validate.

        """
        raise NotImplementedError

    @abstractmethod
    def update(self, mm_desc_ifc, epilog_input):
        """
        Update the provided epilog input.

        Args:
            mm_desc_ifc: The MM descriptor to update, provided as a MatmulDescInterface
                object. epilog_input: The epilog input to validate.

        """
        raise NotImplementedError


@runtime_checkable
class EpilogOutputHandler(Protocol):
    """
    Protocol for epilog handler output validation and setting the appropriate MM descriptor
    attributes.
    """

    @abstractmethod
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """
        The name of the epilog output that is handled (relu_aux, gelu_aux, bgrad, bragda,
        bgradb, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def attributes(self):
        """
        The shape, stride, and dtype name of the output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, mm_desc_ifc):
        """
        Update all the attributes for this epilog, except the pointer.

        Args:
            mm_desc_ifc: The MM descriptor to update, provided as a MatmulDescInterface
            object.
        """
        raise NotImplementedError

    @abstractmethod
    def update_ptr(self, mm_desc_ifc, ptr):
        """
        Set the pointer for this epilog.

        Args:
            mm_desc_ifc: The MM descriptor to update, provided as a MatmulDescInterface
                object.

            ptr: The pointer to set.

        """
        raise NotImplementedError


class BiasHandler(EpilogInputHandler):
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        self.logger = logger
        self.mm_traits = mm_traits

        assert enumerator in [
            Epilog.BIAS,
            Epilog.RELU_BIAS,
            Epilog.RELU_AUX_BIAS,
            Epilog.GELU_BIAS,
            Epilog.GELU_AUX_BIAS,
        ], "Internal error."
        self.enumerator = enumerator

        self.d_dtype_name = d_dtype_name

        self._name = "bias"

    @property
    def name(self):
        return self._name

    def validate(self, bias_tensor):
        # The bias_tensor must be of rank 1, or rank 2 with (M, 1).

        bias_shape = list(bias_tensor.shape)
        bias_strides = list(bias_tensor.strides)

        mm_traits = self.mm_traits

        # Determine the rank of bias_tensor input.
        bias_batch_shape, bias_mm_shape = bias_shape[:-2], bias_shape[-2:]
        bias_mm_strides = bias_strides[-2:]
        if len(bias_mm_shape) == 1:
            s, d = bias_mm_shape[0], bias_mm_strides[0]
            bias_mm_shape += [1]
            bias_mm_strides += [s * d]
        self.logger.debug(f"The MM shape for the bias is {bias_mm_shape} with strides {bias_mm_strides}.")

        Mb, Nb = bias_mm_shape
        if Mb != mm_traits.d_layout.shape[0]:
            raise ValueError(f"The M dimension of the bias vector ({Mb}) must match the local M dimension of A.")

        if Nb != 1:
            raise ValueError(f"The N dimension of the bias vector ({Nb}) must be equal to 1.")

        # Check if the bias_tensor batch shape and axis order match that of the MM, and it's
        # tileable.
        if len(bias_batch_shape) > 0:
            raise ValueError("Batched matmul is currently not supported.")

        if bias_mm_strides[0] != 1:
            raise ValueError(
                f"The stride of the bias {bias_strides} must be 1 along the dimension {len(bias_strides) - 2}, "
                f"which corresponds to the M dimension."
            )

    def update(self, mm_desc_ifc, bias_tensor):
        # Set the bias pointer.
        mm_desc_ifc.bias_pointer = bias_tensor.data_ptr
        # Set the bias data type.
        if bias_tensor.dtype != self.d_dtype_name:
            mm_desc_ifc.bias_data_type = typemaps.NAME_TO_DATA_TYPE[bias_tensor.dtype]


def round_up(m, base):
    return m + (base - m) % base


def relu_aux_mm_shape(m, n):
    """
    Return the RELU auxiliary bitmask matrix shape when stored as uint8 and M is padded to
    128-bit/16-byte multiples.
    """
    # Store bitflag mask using int8 dtype, padded to (128//8 ==) 16 bytes.
    m = round_up(math.ceil(m / 8), base=16)
    return m, n


def gelu_aux_mm_shape(m, n):
    """
    Return the GELU auxiliary matrix shape when M is padded to 8-byte multiples.
    """
    m = round_up(m, base=8)
    return m, n


class ReluAuxHandler(EpilogOutputHandler):
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        self.logger = logger
        self.mm_traits = mm_traits

        assert enumerator in [Epilog.RELU_AUX, Epilog.RELU_AUX_BIAS], "Internal error."
        self.enumerator = enumerator

        self._name = "relu_aux"

        m, n = relu_aux_mm_shape(*mm_traits.d_layout.shape)
        batch_len = len(mm_traits.batch_axis_order)

        self.aux_shape = mm_traits.batch_shape + [m, n]
        aux_axis_order = [batch_len, batch_len + 1] + list(mm_traits.batch_axis_order)  # Column order for the bitmask.
        self.aux_strides = calculate_strides(self.aux_shape, aux_axis_order)
        if aux_dtype_name is not None:
            raise ValueError("Custom type for auxiliary outputs is not supported for RELU epilogs.")
        self.aux_dtype_name = "uint8"

        # We store bitmask using int8 dtype but the values below are in number of elements.
        self.aux_ld = m * 8  # should be consistent with order (currently COL).
        self.aux_batch_offset = m * 8 * n

    @property
    def name(self):
        return self._name

    def attributes(self):
        return self.aux_shape, self.aux_strides, self.aux_dtype_name

    def update(self, mm_desc_ifc):
        # Set the aux LD.
        mm_desc_ifc.epilogue_aux_ld = self.aux_ld
        # Set the pointer to 0x1 to bypass the cuBLAS check.
        mm_desc_ifc.epilogue_aux_pointer = 0x1

    def update_ptr(self, mm_desc_ifc, ptr):
        # Set the aux pointer.
        mm_desc_ifc.epilogue_aux_pointer = ptr


class GeluAuxHandler(EpilogOutputHandler):
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        self.logger = logger
        self.mm_traits = mm_traits

        assert enumerator in [Epilog.GELU_AUX, Epilog.GELU_AUX_BIAS], "Internal error."
        self.enumerator = enumerator

        self.d_dtype_name = d_dtype_name

        self._name = "gelu_aux"

        m, n = gelu_aux_mm_shape(*mm_traits.d_layout.shape)
        batch_len = len(mm_traits.batch_axis_order)

        self.aux_shape = mm_traits.batch_shape + [m, n]
        aux_axis_order = [batch_len, batch_len + 1] + list(mm_traits.batch_axis_order)  # Column order for the GELU inputs.
        self.aux_strides = calculate_strides(self.aux_shape, aux_axis_order)

        if aux_dtype_name:
            self.aux_dtype_name = aux_dtype_name
        else:
            self.aux_dtype_name = c_dtype_name if "float8" in d_dtype_name else d_dtype_name

        self.aux_ld = m  # should be consistent with order (currently COL).
        self.aux_batch_offset = m * n

    @property
    def name(self):
        return self._name

    def attributes(self):
        return self.aux_shape, self.aux_strides, self.aux_dtype_name

    def update(self, mm_desc_ifc):
        # Set the aux LD.
        mm_desc_ifc.epilogue_aux_ld = self.aux_ld
        # Set the pointer to 0x1 to bypass the cuBLAS check.
        mm_desc_ifc.epilogue_aux_pointer = 0x1
        if self.aux_dtype_name is not None:
            mm_desc_ifc.epilogue_aux_data_type = typemaps.NAME_TO_DATA_TYPE[self.aux_dtype_name]

    def update_ptr(self, mm_desc_ifc, ptr):
        # Set the aux pointer.
        mm_desc_ifc.epilogue_aux_pointer = ptr


class BgradHandler(EpilogOutputHandler):
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        self.logger = logger
        self.mm_traits = mm_traits

        assert enumerator in [Epilog.DRELU_BGRAD, Epilog.DGELU_BGRAD, Epilog.BGRADA, Epilog.BGRADB], "Internal error."
        self.enumerator = enumerator

        self._name = enumerator.name.lower()

        if aux_dtype_name is not None:
            raise ValueError("Custom type for auxiliary outputs is not supported for BGRAD epilogs.")

        # For BGRADB with BIAS_RESULT_SCHEME=FULL the cuBLASMp output is always replicated.
        m = mm_traits.N if enumerator == Epilog.BGRADB else mm_traits.d_layout.shape[0]
        batch_len = len(mm_traits.batch_axis_order)

        self.bgrad_shape = shape = [m]
        if mm_traits.batch_shape:
            shape = shape + [1]
            self.bgrad_shape = mm_traits.batch_shape + self.bgrad_shape + [1]

        bgrad_axis_order = [batch_len + a for a in range(len(shape))] + list(mm_traits.batch_axis_order)  # Column order.
        self.bgrad_strides = calculate_strides(self.bgrad_shape, bgrad_axis_order)

        self.d_dtype_name = d_dtype_name
        self.bgrad_dtype_name = d_dtype_name
        self.bgrad_batch_offset = m

    @property
    def name(self):
        return self._name

    def attributes(self):
        return self.bgrad_shape, self.bgrad_strides, self.bgrad_dtype_name

    def update(self, mm_desc_ifc):
        # The bgrad data type is by default the data type of the result for all the cases we
        # support.
        assert self.bgrad_dtype_name == self.d_dtype_name, "Internal error."

    def update_ptr(self, mm_desc_ifc, ptr):
        # Set the bgrad pointer.
        mm_desc_ifc.bias_pointer = ptr


class DGeluAuxHandler(EpilogInputHandler):
    def __init__(self, logger, mm_traits, enumerator, c_dtype_name, d_dtype_name, aux_dtype_name):
        self.logger = logger
        self.mm_traits = mm_traits

        assert enumerator in [Epilog.DGELU, Epilog.DGELU_BGRAD], "Internal error."
        self.enumerator = enumerator

        self.d_dtype_name = d_dtype_name

        self.batch_offset = None

        self._name = "gelu_aux"

        # The GELU aux matrix shape, including padding.
        self.mm_m, self.mm_n = gelu_aux_mm_shape(*mm_traits.d_layout.shape)

        self.aux_ld = self.mm_m  # should be consistent with order (currently COL).

        # K=1 is not supported by cuBLAS
        if mm_traits.K == 1:
            raise ValueError("K=1 is not supported for DGELU epilogs")

    @property
    def name(self):
        return self._name

    def validate(self, gelu_aux_tensor):
        # The gelu_aux_tensor must be of rank 2 or its batched version of the latter (...,
        # M, N).
        gelu_aux_shape = list(gelu_aux_tensor.shape)
        gelu_aux_strides = list(gelu_aux_tensor.strides)

        # The dtype must be the same as that of D.
        if gelu_aux_tensor.dtype != self.d_dtype_name:
            raise ValueError(
                f"The dtype of the GELU auxiliary input for epilog {self.enumerator.name} must be '{self.d_dtype_name}'. "
                f"The epilog input's dtype is '{gelu_aux_tensor.dtype}'."
            )

        mm_traits = self.mm_traits

        # Determine the rank of gelu_aux_tensor input.
        gelu_aux_batch_shape, gelu_aux_mm_shape = gelu_aux_shape[:-2], gelu_aux_shape[-2:]
        gelu_aux_batch_strides, gelu_aux_mm_strides = gelu_aux_strides[:-2], gelu_aux_strides[-2:]

        # The MM shape must match, the MM must be in col order, and the batch order must
        # match.
        Ma, Na = gelu_aux_mm_shape
        if Ma != self.mm_m or Na != self.mm_n:
            raise ValueError(
                f"The auxiliary epilog input for epilog {self.enumerator.name} must have "
                f"the MM shape (..., {self.mm_m}, {self.mm_n}). The epilog input's MM shape is (..., {Ma}, {Na})."
            )

        # Check if the gelu_aux_tensor batch shape and axis order match that of the MM, and
        # it's tileable.
        if len(gelu_aux_batch_shape) > 0:
            if gelu_aux_batch_shape != mm_traits.batch_shape:
                raise ValueError(
                    f"The batch dimensions of the GELU auxiliary input {gelu_aux_batch_shape} must match with that "
                    f"of the matrix multiplication definition {mm_traits.batch_shape}."
                )

            if (gelu_aux_batch_axis_order := axis_order_in_memory(gelu_aux_batch_strides)) != mm_traits.batch_axis_order:
                raise ValueError(
                    f"The batch axis order of the GELU auxiliary input {gelu_aux_batch_axis_order} "
                    f"must match with that of the other operands {mm_traits.batch_axis_order}."
                )

            if not check_batch_tileable(gelu_aux_batch_shape, gelu_aux_batch_strides):
                message = (
                    f"The batch layout for GELU auxiliary input corresponding to shape = {gelu_aux_batch_shape} and "
                    f"strides = {gelu_aux_batch_strides} is currently not supported because it is not tileable."
                )
                raise ValueError(message)

        if gelu_aux_mm_strides[0] != 1:
            raise ValueError(
                f"The stride of the GELU auxiliary input {gelu_aux_strides} must be 1 "
                f"along the dimension {len(gelu_aux_strides) - 2}, which corresponds to the M dimension."
            )

        self.batch_offset = min(gelu_aux_batch_strides) if gelu_aux_batch_strides else 0  # gelu_aux broadcast

        if self.batch_offset > 0:
            assert self.batch_offset >= self.mm_m * self.mm_n, "Tensor data must not overlap."

    def update(self, mm_desc_ifc, gelu_aux_tensor):
        # Set the epilog aux pointer.
        mm_desc_ifc.epilogue_aux_pointer = gelu_aux_tensor.data_ptr
        # Set the aux LD.
        mm_desc_ifc.epilogue_aux_ld = self.aux_ld
        # Set the gelu aux data type.
        if gelu_aux_tensor.dtype != self.d_dtype_name:
            mm_desc_ifc.epilogue_aux_data_type = typemaps.NAME_TO_DATA_TYPE[gelu_aux_tensor.dtype]


EPILOG_INPUT_HANDLERS_MAP: dict[cublasMp.MatmulEpilogue, list[type[EpilogInputHandler]]] = {
    Epilog.ALLREDUCE: [],
    Epilog.RELU: [],
    Epilog.RELU_AUX: [],
    Epilog.GELU: [],
    Epilog.GELU_AUX: [],
    Epilog.BIAS: [BiasHandler],
    Epilog.RELU_BIAS: [BiasHandler],
    Epilog.RELU_AUX_BIAS: [BiasHandler],
    Epilog.GELU_BIAS: [BiasHandler],
    Epilog.GELU_AUX_BIAS: [BiasHandler],
    Epilog.DGELU: [DGeluAuxHandler],
    Epilog.DGELU_BGRAD: [DGeluAuxHandler],
    Epilog.BGRADA: [],
    Epilog.BGRADB: [],
}

EPILOG_OUTPUT_HANDLERS_MAP: dict[cublasMp.MatmulEpilogue, list[type[EpilogOutputHandler]]] = {
    Epilog.ALLREDUCE: [],
    Epilog.RELU: [],
    Epilog.RELU_AUX: [ReluAuxHandler],
    Epilog.GELU: [],
    Epilog.GELU_AUX: [GeluAuxHandler],
    Epilog.BIAS: [],
    Epilog.RELU_BIAS: [],
    Epilog.RELU_AUX_BIAS: [ReluAuxHandler],
    Epilog.GELU_BIAS: [],
    Epilog.GELU_AUX_BIAS: [GeluAuxHandler],
    Epilog.DGELU: [],
    Epilog.DGELU_BGRAD: [BgradHandler],
    Epilog.BGRADA: [BgradHandler],
    Epilog.BGRADB: [BgradHandler],
}

EPILOG_MINIMUM_VERSIONS_MAP: dict[cublasMp.MatmulEpilogue | None, dict[str, int]] = {
    None: {"cublasMp": 00000},
    Epilog.ALLREDUCE: {"cublasMp": 800},
    Epilog.RELU: {"cublasMp": 800},
    Epilog.GELU: {"cublasMp": 800},
    Epilog.BIAS: {"cublasMp": 800},
    Epilog.RELU_AUX: {"cublasMp": 800},
    Epilog.GELU_AUX: {"cublasMp": 800},
    Epilog.RELU_BIAS: {"cublasMp": 800},
    Epilog.RELU_AUX_BIAS: {"cublasMp": 800},
    Epilog.GELU_BIAS: {"cublasMp": 800},
    Epilog.GELU_AUX_BIAS: {"cublasMp": 800},
    Epilog.DGELU: {"cublasMp": 800},
    Epilog.BGRADA: {"cublasMp": 800},
    Epilog.BGRADB: {"cublasMp": 800},
    Epilog.DGELU_BGRAD: {"cublasMp": 800},
}
