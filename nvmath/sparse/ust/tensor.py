# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the universal sparse tensor (UST).
"""

__all__ = ["Tensor"]

import contextlib
import re
from typing import Literal

import numpy as np
from cuda import pathfinder

from nvmath._internal.layout import is_contiguous_and_dense
from nvmath.internal import tensor_wrapper, utils
from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc_numpy import NumpyTensor
from nvmath.sparse._internal.common_utils import sparse_or_dense, wrap_sparse_operand

from ._converters import TensorConverter
from ._drawer import animate_tensor, draw_tensor, draw_tensor_raw, draw_tensor_storage
from ._emitter import emit_apply, populate_apply_parameters
from ._jit import compile_cpp_and_link, compile_python_function, launch_kernel
from ._utils import LevelMap
from .tensor_format import Dimension, LevelFormat, TensorFormat, is_unique


def _tensor_to_list(wrapped_operand, truncate=32):
    m = wrapped_operand.memory_buffer()
    t = m.tensor
    o = [str(v.item()) for v in t[:truncate]]
    if m.size > truncate:
        o += ["..."]

    if m.size > 2 * truncate:
        o += [str(v.item()) for v in t[-truncate:]]

    return "[{}]".format(", ".join(o))


def _top_array(value, *, dtype, device_id, dense_tensorholder_type, stream_holder):
    a = utils.create_empty_tensor(dense_tensorholder_type, (2,), dtype, device_id, stream_holder, verify_strides=False)
    a.tensor[0], a.tensor[1] = 0, value
    return a


def _convert_ndbuffer_perhaps(wrapped_operand):
    package = utils.infer_object_package(wrapped_operand.tensor)
    if package != "nvmath":
        return wrapped_operand

    package = "cuda"
    if wrapped_operand.device_id == "cpu":
        stream = None
    else:
        stream = utils.get_or_create_stream(wrapped_operand.device_id, None, package)

    return NumpyTensor.create_host_from(wrapped_operand, stream)


def _axis_order_in_memory(shape, strides):
    """
    Compute the order in which the axes appear in memory.
    """
    # Scalars.
    if len(shape) == 0:
        return []

    # The shape is used to resolve cases like (1, 2, 1) : (2, 1, 1) in CuTe notation.
    # Use numpy's lexsort for better performance (avoids creating intermediate tuples)
    ndim = len(strides)
    axis_order = np.lexsort((np.arange(ndim), shape, strides))
    return tuple(axis_order)


def _calculate_strides(shape, axis_order):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [None] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides


def _dense_ust_axis_order(tensor_format, shape):
    """
    Compute the strides for a dense contiguous N-D tensor represented as a UST.
    """
    dimensions = tensor_format.dimensions
    level_keys = list(tensor_format.levels.keys())
    axis_order = [None] * len(dimensions)
    for d, label in enumerate(dimensions):
        index = level_keys.index(label)
        axis_order[index] = d

    return list(reversed(axis_order))


def _get_smallest_index_type(index):
    dtypes = [np.int32, np.int64]
    for dtype in dtypes:
        try:
            dtype(index)
            break
        except OverflowError:
            pass

    return np.dtype(dtype).name


SHARED_UST_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_UST_DOCUMENTATION.update(
    {
        #
        "extents": "The extents (shape) of the sparse tensor as a Python sequence",
        #
        "tensor_format": "The sparse tensor format as a :class:`TensorFormat` object.",
        #
        "index_type": """\
The index type of the tensor as a NumPy-recognizable string or CudaDataType.""".replace("\n", " "),
        #
        "dtype": """\
The datatype of the tensor as a NumPy-recognizable string or CudaDataType.""".replace("\n", " "),
    }
)


@utils.docstring_decorator(SHARED_UST_DOCUMENTATION, skip_missing=False)
class Tensor:
    """
    The universal sparse tensor binds extents, data type, index type, etc with a universal
    sparse tensor format object.

    .. note::  The constructor is currently private and for internal use only. Users should
       use any of the methods :meth:`Tensor.from_package`, :meth:`Tensor.from_file`, or
       :meth:`Tensor.convert` to construct a UST.


    Args:
        extents: {extents}
        tensor_format: {tensor_format}
        index_type: {index_type}
        dtype: {dtype}
        package: The package using which the dense tensors underlying the sparse
                 representation will be allocated. It must be a Python module object.
    """

    def __init__(
        self, extents, *, tensor_format=None, index_type="int32", dtype="float32", package=None, options=None, stream=None
    ):
        if tensor_format is None:
            raise TypeError("The tensor format must be specified.")
        if not isinstance(tensor_format, TensorFormat):
            raise TypeError(f"The tensor format {tensor_format} must be a TensorFormat object.")
        if len(extents) != tensor_format.num_dimensions:
            raise TypeError(
                f"The tensor rank {len(extents)} must equal the format dimensionality \
{tensor_format.num_dimensions}."
            )
        if np.prod(extents) <= 0:
            raise TypeError(f"The tensor extents {extents} do not define any elements.")

        assert options is None, "Internal error."

        self._extents = list(extents)
        self._levels = tensor_format.dim2lvl(extents, True)
        self._tensor_format = tensor_format
        self._index_type = index_type
        self._dtype = dtype
        self._package = package

        # The UST uses a dictionary from "levels" to the package arrays for the positions
        # and coordinates, and a direct array for values.
        self._pos = LevelMap()
        self._crd = LevelMap()
        self._val = None

        # Set wrapped operand.
        self.wrapped_operand = None

        # Kernel.
        self._kernel = None
        self._with_indices = False

    @classmethod
    def from_file(cls, filename, stream=None):
        """
        A helper to create a universal sparse tensor (in COO representation) from a file.
        The supported file formats are the Matrix Market format (.mtx) for matrices and
        the FROSTT format (.tns) for tensors.
        """
        message = f"The format of {filename} is not recognized. The supported extensions are '.mtx' and '.tns'"
        m = re.match(r".*\.(\w+)", filename)
        if m is None:
            raise TypeError(message)
        suffix = m.group(1)

        if suffix == "mtx":
            # As of version 1.12, scipy.io.mmread is based on fast_matrix_market. This
            # provides the fastest way to read from the Matrix Market and simplifies
            # getting the COO in canonical format.

            try:
                import scipy as sp
            except ImportError:
                raise ImportError("SciPy is required for this functionality.") from None

            coo = sp.io.mmread(filename, spmatrix=True)
            coo.sum_duplicates()
            return cls.from_package(coo)

        if suffix == "tns":
            # Convenience Python utility to map file data to indices and values. Note
            # that we could use this map to construct the UST directly as an unordered
            # and nonunique COO format. However, we go through torch tensor to simplify
            # getting the COO in canonical format.
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch is required for this functionality.") from None

            data = np.genfromtxt(filename, comments="#", unpack=True, dtype=None)
            num_dimensions = len(data) - 1

            if num_dimensions < 1:
                raise ValueError(f"cannot parse {filename} with FROSTT structure")

            extents = tuple(max(data[d]) for d in range(num_dimensions))
            indices = torch.stack([torch.as_tensor(data[d] - 1) for d in range(num_dimensions)], dim=0)
            values = data[num_dimensions]

            coo = torch.sparse_coo_tensor(indices, values, extents)
            return cls.from_package(coo.coalesce())

        raise TypeError(message)

    @classmethod
    def from_package(cls, tensor, stream=None):
        """
        Create an universal sparse tensor from a sparse package tensor. This is a
        zero-copy operation where the UST shares a view of the sparse data with the
        sparse tensor. A strided tensor (like :class:`numpy.ndarray` or
        :class:`torch.Tensor`) can also be viewed as a UST, with the current limitation
        that there are no "holes" (the strided tensor is dense, in other words).

        The currently supported libraries are NumPy, SciPy, CuPy, and PyTorch and
        both CPU and GPU (CUDA) memory spaces are supported.

        Args:
            tensor: The source sparse tensor from NumPy, SciPy, CuPy, or PyTorch packages.
            stream: {stream}

        Returns:
            A UST view of the sparse or dense package tensor.
        """
        # Note: since the API takes in a stream object, it's the user's responsibility
        # to ensure ordering of operations wrt to the provided stream. If we decide to use
        # a package stream for UST, then we will internally order it wrt the user stream.
        # The `wrapped.operand.to_ust()` method does this if a stream pointer is provided.

        tensor_type = sparse_or_dense(tensor)
        if tensor_type == "sparse":
            try:
                wrapped_operand = wrap_sparse_operand(tensor)
                # If we decide to use a package stream for UST, then we will internally
                # order it wrt the user stream. The `wrapped.operand.to_ust()` method does
                # this if a stream pointer is provided.
                device_id = wrapped_operand.device_id
                device_ctx = contextlib.nullcontext() if device_id == "cpu" else utils.device_ctx(device_id)
                with device_ctx:
                    ust = wrapped_operand.to_ust(stream=stream)
                ust.wrapped_operand = wrapped_operand
                return ust
            except Exception as e:
                raise TypeError(f"The sparse operand type {type(tensor)} is unsupported or invalid.") from e

        # If we decide to use a package stream for UST, then we will internally order
        # it wrt the user stream. The `synchronize_dense_tensor` function does this
        # for dense operands.

        assert tensor_type == "dense", "Internal error."

        return cls._from_dense(tensor)

    def _is_dense_format(self):
        """
        Check if the UST representation is a dense strided layout.
        """
        # TODO: Use the level spec instead of inferring from the name.
        return self.tensor_format.name.startswith("Scalar") or self.tensor_format.name.startswith("Dense")

    def _dense_tensor_from_buffer(self):
        """
        Create a view of the buffer as a dense tensor (only applicable to strided layout).
        """
        shape = self.extents
        axis_order = _dense_ust_axis_order(self.tensor_format, shape)
        strides = _calculate_strides(shape, axis_order)

        wrapped_dense = self.val.memory_buffer_to_tensor(shape, strides)  # unflatten

        return wrapped_dense

    def to_package(self, options=None, stream=None):
        """
        Create a sparse package tensor from the universal sparse tensor. This is a
        zero-copy operation where the UST shares a view of the sparse data with the package
        tensor.

        Returns:
            The sparse or dense package tensor view into this UST.
        """
        device_id = self.device_id
        device_ctx = contextlib.nullcontext() if device_id == "cpu" else utils.device_ctx(device_id)
        with device_ctx:
            if self._is_dense_format():
                # Extract from the wrapped operand, if it exists.
                if self.wrapped_operand is not None:
                    return self.wrapped_operand.tensor

                # Otherwise, create a new dense tensor view.
                return self._dense_tensor_from_buffer().tensor

            return self.wrapped_operand.to_package()

    def convert(self, *, tensor_format=None, index_type=None, dtype=None):
        """
        Convert a UST into a new UST with the specified format, index type, and data
        type. The default values of these for the target UST are taken from the
        corresponding ones in the source UST.

        Please note that this is a proof-of-concept implementation that supports
        converting any UST format into any other UST format. Even though some
        fast-path solutions are provided, the current general solution in
        :class:`TensorConverter` has not been optimized for speed yet.

        .. note:: The target shares a source kernel only for trivial conversions.

        Args:
            tensor_format: {tensor_format}
            index_type: {index_type}
            dtype: {dtype}

        Returns:
            A UST in the specified format, with the specified index and data types.
        """
        if tensor_format is None:
            tensor_format = self.tensor_format
        if index_type is None:
            index_type = self.index_type
        if dtype is None:
            dtype = self.dtype

        # Fast path.
        if self.tensor_format.name == tensor_format.name and self.index_type == index_type and self.dtype == dtype:
            return self.clone()

        target = Tensor(
            self.extents,
            tensor_format=tensor_format,
            index_type=index_type,
            dtype=dtype,
        )
        TensorConverter(self, target).run()

        # Set wrapped operand.
        if target._is_dense_format():
            target.wrapped_operand = target._dense_tensor_from_buffer()
        else:
            target.wrapped_operand = wrap_sparse_operand(target)

        return target

    def to(self, device_id: int | Literal["cpu"], stream=None):
        """Copy the UST to a different device (contents only).

        The sparse representation is not copied (it is a  view) if the UST
        is already on the requested device.

        .. note:: Any kernel associated with the source is not copied over to target.

        Args:
            device_id: The CUDA device ordinal, or "cpu".
            stream: {stream}

        Returns:
            A copy of the UST on the specified device.
        """
        package = self._dense_tensorholder_type.name

        if (device_id == "cpu") != (self.device_id == "cpu") and package != "torch":
            raise NotImplementedError(
                f"The to() operation from memory space {self.device_id} to memory space {device_id} \
currently requires the UST to be backed by PyTorch storage."
            )

        if (
            isinstance(device_id, int)
            and isinstance(self.device_id, int)
            and device_id != self.device_id
            and package != "torch"
        ):
            raise ValueError(
                "The to() operation currently supports copying across different devices only for \
a UST backed by PyTorch storage."
            )

        # The device for the stream should be the source device. If the source is CPU, the
        # operation is blocking and the stream holder can be None.
        stream_holder = None
        if self.device_id != "cpu":
            # For internal use, we accept StreamHolder so that UST `to` has a
            # consistent interface with `TensorHolder.to`.
            if isinstance(stream, StreamHolder):
                stream_holder = stream
            else:
                stream_holder = utils.get_or_create_stream(self.device_id, stream, package)

        target = Tensor(
            self.extents,
            tensor_format=self.tensor_format,
            index_type=self.index_type,
            dtype=self.dtype,
            package=self._package,
            options=None,
            stream=stream,
        )

        # Set the UST data.
        target._pos = self._pos.to(device_id, stream_holder)
        target._crd = self._crd.to(device_id, stream_holder)
        assert self._val is not None, "Internal error: UST is only partially constructed."
        target._val = self._val.to(device_id, stream_holder)

        # Set wrapped operand.
        if target._is_dense_format():
            target.wrapped_operand = target._dense_tensor_from_buffer()
        else:
            target.wrapped_operand = wrap_sparse_operand(target)

        return target

    def copy_(self, src, stream=None):
        """
        Copy a compatible UST into this one, where compatible means that the source
        UST has the same shape, tensor format, non-zero structure, index type, and data
        type as this one.

        .. note:: Any kernel associated with the source is not copied over to target, and
            any kernel that was associated with this one is reset to None.

        Args:
            src: The source UST (:class:`nvmath.sparse.ust.Tensor`) that will be
                copied into this one.
            stream: {stream}
        """

        if self.tensor_format.name != src.tensor_format.name:
            raise ValueError("The source and target formats are not compatible.")

        if self.extents != src.extents:
            raise ValueError(f"The source {src.extents} and target {self.extents} extents are not compatible.")

        if self.index_type != src.index_type:
            raise ValueError(f"The source {src.index_type} and target {self.index_type} index types are not compatible.")

        if self.dtype != src.dtype:
            raise ValueError(f"The source {src.dtype} and target {self.dtype} dtypes are not compatible.")

        stream_holder = None
        device_id = src.device_id
        if device_id != "cpu":
            # For internal use, we accept StreamHolder so that UST `copy_` has a
            # consistent interface with `TensorHolder.copy_`.
            if isinstance(stream, StreamHolder):
                stream_holder = stream
            else:
                package = self._dense_tensorholder_type.name
                stream_holder = utils.get_or_create_stream(device_id, stream, package)

        # Copy the UST data.
        self._pos.copy_(src._pos, stream_holder)
        self._crd.copy_(src._crd, stream_holder)
        assert self._val is not None, "Internal error: UST is only partially constructed."
        self._val.copy_(src._val, stream_holder)

        self._kernel = None
        self._with_indices = False

    def clone(self, stream=None):
        """
        Clone the UST. The sparse representation of the UST is copied to the target, while
        any associated kernel is shared between source and target.

        .. note:: Any kernel associated with the source is shared with the target.

        Args:
            stream: {stream}

        Returns:
            A clone of this UST.
        """
        stream_holder = None
        device_id = self.device_id
        if device_id != "cpu":
            package = self._dense_tensorholder_type.name
            stream_holder = utils.get_or_create_stream(device_id, stream, package)

        target = Tensor(
            self.extents,
            tensor_format=self.tensor_format,
            index_type=self.index_type,
            dtype=self.dtype,
            package=self._package,
            options=None,
            stream=stream,
        )

        # Empty UST data.
        target._pos = self._pos.empty_like(stream_holder)
        target._crd = self._crd.empty_like(stream_holder)
        target._val = utils.create_empty_tensor(
            self._dense_tensorholder_type, self.nse, self.dtype, device_id, stream_holder, verify_strides=False
        )

        # Copy the UST data.
        target._pos.copy_(self._pos, stream_holder)
        target._crd.copy_(self._crd, stream_holder)
        target._val.copy_(self._val, stream_holder)

        target._kernel = self._kernel  # not deeply cloned
        target._with_indices = self._with_indices

        # Set wrapped operand.
        if target._is_dense_format():
            target.wrapped_operand = target._dense_tensor_from_buffer()
        else:
            target.wrapped_operand = wrap_sparse_operand(target)

        return target

    def set_kernel(self, user_code, *, with_indices=True, arch=None):
        """
        Set the callback code (a unary Python function, possibly already compiled into
        LTO-IR) and link this with the actual traversal kernel for the UST. This can be
        used to apply an user-defined transformation to each element of the UST, where the
        user-defined transformation can potentially depend on the coordinates (indices)
        of the element.

        Args:
            code: The callback code as a unary Python function object or as a bytes object
               with the LTO-IR code. The function must take a single argument representing
               the original value if ``with_indices`` is ``False``, otherwise it takes
               the original value followed by the coordinate for that value, as a sequence.
               The original value must be representable in the data type of the UST, while
               the indices must be integers.

            with_indices: A flag to choose between the value-only or value-with-indices
                signatures.

        .. tip:: If the indices (coordinate) are not needed for the user-defined
            transformation, it is much more efficient to set ``with_indices=False``.
        """

        if self._val.device == "cpu":
            raise ValueError("A CUDA kernel cannot be specified for the CPU memory space.")

        if not isinstance(user_code, bytes):
            signature = self.dtype, self.dtype  # return type, value dtype
            if with_indices:
                signature += (self.index_type,) * self.num_dimensions
            user_code = compile_python_function(user_code, "apply", signature)

        driver_src = emit_apply(self, with_indices)

        complex_path = pathfinder.find_nvidia_header_directory("cccl")
        narrow_prec_path = pathfinder.find_nvidia_header_directory("cudart")
        compiler_options = {
            "std": "c++17",
            "link_time_optimization": True,
            "arch": arch,
            "include_path": [complex_path, narrow_prec_path],
        }
        linker_options = {"link_time_optimization": True}

        self._kernel = compile_cpp_and_link(
            src_code=driver_src,
            object_code=user_code,
            function_name="apply_kernel",
            compiler_options=compiler_options,
            linker_options=linker_options,
        )
        self._with_indices = with_indices

    def _check_valid_kernel(self, *args, **kwargs):
        """
        Check if a kernel has been set.
        """
        if not self._kernel:
            raise ValueError("A kernel has not been set using `set_kernel`.")

    @utils.precondition(_check_valid_kernel)
    def run_kernel(self, stream=None):
        """
        Run the kernel associated with this UST object.
        """

        assert self._kernel is not None, "Internal error."

        parameters, size = populate_apply_parameters(self, self._with_indices)

        package = self.val.__class__.name
        device_id = self.val.device_id

        stream_holder = utils.get_or_create_stream(device_id, stream, package)
        launch_kernel(
            self._kernel, parameters, problem_size=size, device_id=device_id, stream_holder=stream_holder, blocking=True
        )

    def get_value(self, indices):
        """
        Retrieve the value at specified indices (coordinate) or ``None`` if there is no
        value at that coordinate. This is *not* random access, and searches compressed
        levels.

        Returns:
          ``tensor.get_value[i, j, k, ...]` returns the value of the element appearing at
          logical dimension indices ``i, j, k, ...` or ``None`` if there is no element at
          that coordinate.
        """
        if len(indices) != self.num_dimensions:
            raise ValueError(
                f"The number of indices specified {indices} doesn't match the number of \
tensor dimensions {self.num_dimensions}."
            )

        p = self._locate(list(self.tensor_format.levels.values()), self.tensor_format.dim2lvl(indices), 0, 0)
        return p if p is None else self.val.tensor[p]

    def draw(self, name=None):
        """
        Draws the tensor contents (for 1D, 2D, 3D). The method saves the constructed
        :class:`PIL.Image` in a file named `name` when name is not None. It always
        returns the :class:`PIL.Image` object directly as well, so that the caller
        can directly manipulate the constructed image in some other manner.

        This method is useful to illustrate tensor contents of smaller examples.

        Args:
            name: filename to save the image to, if not None
        Returns:
            PIL.Image, can be displayed with show()

        Examples:
            >>> import scipy.sparse as sp
            >>> from nvmath.sparse.ust import Tensor

            Create a sparse scipy matrix in CSR format.

            >>> a = sp.random_array((8, 8), density=0.1, format="csr", dtype="float64")

            Convert the scipy matrix to UST.

            >>> u = Tensor.from_package(a)

            Obtain the tensor contents as image (and e.g. display as ``img.show()``).

            >>> img = u.draw()  # doctest: +SKIP
        """
        return draw_tensor(self, name)

    def draw_storage(self, name=None):
        """
        Draws the tensor storage (for any UST). The method saves the constructed
        :class:`PIL.Image` in a file named `name` when name is not None. It always
        returns the :class:`PIL.Image` object directly as well, so that the caller
        can directly manipulate the constructed image in some other manner.

        This method is useful to illustrate the UST storage of smaller examples.

        Args:
            name: filename to save the image to, if not None
        Returns:
            PIL.Image, can be displayed with show()

        Examples:
            >>> import scipy.sparse as sp
            >>> from nvmath.sparse.ust import Tensor

            Create a sparse scipy matrix in CSR format.

            >>> a = sp.random_array((8, 8), density=0.1, format="csr", dtype="float64")

            Convert the scipy matrix to UST.

            >>> u = Tensor.from_package(a)

            Obtain the tensor storage as image (and e.g. display as ``img.show()``).

            >>> img = u.draw_storage()  # doctest: +SKIP
        """
        return draw_tensor_storage(self, name)

    def draw_raw(self, name=None):
        """
        Draws the tensor nonzero structure (for 2D, 3D). The method saves the constructed
        :class:`PIL.Image` in a file named `name` when name is not None. It always
        returns the :class:`PIL.Image` object directly as well, so that the caller
        can directly manipulate the constructed image in some other manner.

        This method scales to larger tensors.

        Args:
            name: filename to save the image to, if not None
        Returns:
            PIL.Image, can be displayed with show()

        Examples:
            >>> import scipy.sparse as sp
            >>> from nvmath.sparse.ust import Tensor

            Create a sparse scipy matrix in CSR format.

            >>> a = sp.random_array((512, 512), density=0.01, format="csr", dtype="float64")

            Convert the scipy matrix to UST.

            >>> u = Tensor.from_package(a)

            Obtain the tensor nonzero structure as image (e.g. display as ``img.show()``).

            >>> img = u.draw_raw()  # doctest: +SKIP
        """
        return draw_tensor_raw(self, name)

    def animate(self, name=None):
        """
        Animates the tensor nonzero structure (for 3D). The method either saves the
        animated GIF in a file named `name` when name is not None, or otherwise
        returns a :class:`IPython.display.HTML` that can be embedded by the caller.

        This method scales to larger tensors.

        Args:
            name: filename to save the animated GIF to, if not None
        Returns:
            IPython.display.HTML, if name is None (to embed in other output)

        Examples:
            >>> import torch
            >>> from nvmath.sparse.ust import Tensor

            Create a 3-D tensor, as COO.

            >>> a = torch.eye(32).repeat(32, 1, 2).to_sparse()

            Convert the tensor to UST.

            >>> u = Tensor.from_package(a)

            Obtain the animation as HTML.

            >>> html = u.animate()  # doctest: +SKIP
        """
        return animate_tensor(self, name)

    @property
    def extents(self):
        """
        The extents (shape) of this UST object.
        """
        return self._extents

    shape = extents

    @property
    def levels(self):
        """
        The extents corresponding to each level for this UST object.
        """
        return self._levels

    @property
    def size(self):
        """
        The envelope of this UST object (product of the extents).
        """
        # TODO: rename this to envelope or bounding_volume?
        return np.prod(self.extents)

    @property
    def num_dimensions(self):
        """
        The number of dimensions of this UST object.
        """
        return self.tensor_format.num_dimensions

    @property
    def num_levels(self):
        """
        The number of levels of this UST object.
        """
        return self.tensor_format.num_levels

    @property
    def tensor_format(self):
        """
        The UST format as a :class:`nvmath.sparse.ust.TensorFormat` object.
        """
        return self._tensor_format

    @property
    def index_type(self):
        """
        The index type of this UST object as a string.
        """
        return self._index_type

    @property
    def dtype(self):
        """
        The data type of this UST object as a string.
        """
        return self._dtype

    def pos(self, level):
        """
        The positions corresponding to the specified level.

        Args:
            level: An ordinal specifying the level.
        """
        return self._pos.get(level, None)

    def crd(self, level):
        """
        The coordinates corresponding to the specified level.

        Args:
            level: An ordinal specifying the level.
        """
        return self._crd.get(level, None)

    @property
    def val(self):
        """
        The explicit values stored in this UST object.
        """
        return [] if self._val is None else self._val

    @property
    def device(self):
        """
        The memory space of this UST object ("cpu" or "cuda").
        """
        return "cpu" if self._val is None else self._val.device

    @property
    def device_id(self):
        """
        The device ID of this UST object (CUDA ordinal or "cpu").
        """
        return "cpu" if self._val is None else self._val.device_id

    @property
    def nse(self):
        """
        The number of stored elements in this UST object.
        """
        return 0 if self._val is None else self._val.size

    @property
    def base(self):
        """
        Get the "package" tensor from which the UST view was created, if available.
        """
        if self.wrapped_operand is not None:
            return self.wrapped_operand.tensor

    @property
    def _dense_tensorholder_type(self):
        """
        (Internal use only) The tensor holder type for the wrapped dense tensors.
        """
        return self.val.__class__

    def __repr__(self):
        s = (
            f"---- Sparse Tensor<VAL={self.dtype},"
            f"POS={self.index_type},CRD={self.index_type},"
            f"DIM={self.num_dimensions},LVL={self.num_levels}>\n"
        )
        s += f"format   : {self.tensor_format}\n"
        s += f"device   : {self.device}\n"
        s += f"dim      : {self.extents}\n"
        s += f"lvl      : {self.levels}\n"
        s += f"nse      : {self.nse}\n"
        data = 0
        for level in range(self.num_levels):
            pos = self.pos(level)
            if pos is None:
                continue
            pos = _convert_ndbuffer_perhaps(pos)
            s += f"pos[{level}]   : {_tensor_to_list(pos)} #{pos.size}\n"
            data += pos.size * pos.itemsize
        for level in range(self.num_levels):
            crd = self.crd(level)
            if crd is None:
                continue
            crd = _convert_ndbuffer_perhaps(crd)
            s += f"crd[{level}]   : {_tensor_to_list(crd)} #{crd.size}\n"
            data += crd.size * crd.itemsize
        if self.nse > 0:
            val = self.val
            s += f"values   : {_tensor_to_list(val)} #{val.size}\n"
            data += val.size * val.itemsize
        s += f"data     : {data} bytes\n"
        s += f"sparsity : {(100.0 - (100.0 * self.nse) / self.size):.2f}%\n"
        s += "----"
        return s

    def _locate(self, formats, lvls, level, p):
        # Exhausted all levels.
        if level == self.num_levels:
            return p
        # Obtain format and properties at current level.
        v = formats[level]
        if isinstance(v, tuple):
            fmt, prop = v
        else:
            fmt, prop = v, None
        # Handle level format.
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.BATCH or fmt == LevelFormat.RANGE:
            return self._locate(formats, lvls, level + 1, p * self.levels[level] + lvls[level])
        elif fmt == LevelFormat.COMPRESSED:
            unique = is_unique(prop)
            pos = self.pos(level).tensor
            crd = self.crd(level).tensor
            assert pos.ndim == crd.ndim
            adjust = 0
            if pos.ndim > 1:
                # This only happens for prior BATCH. Correct the higher-dimensions in
                # both pos and crd buffers and adjust position by #nnz per batch.
                assert pos.ndim == level
                for i in range(level - 1):
                    assert formats[i] == LevelFormat.BATCH
                    pos = pos[lvls[i]]
                    crd = crd[lvls[i]]
                adjust = pos[-1].item() * (p - lvls[level - 1]) // self.levels[level - 1]
                p = lvls[level - 1]
            lo = pos[p].item()
            hi = pos[p + 1].item()
            for i in range(lo, hi):
                if crd[i] == lvls[level]:
                    cpos = self._locate(formats, lvls, level + 1, i + adjust)
                    if unique:
                        return cpos  # always end scan (unique)
                    elif cpos is not None:
                        return cpos  # only end scan on success (non-unique)
        elif fmt == LevelFormat.SINGLETON:
            crd = self.crd(level).tensor
            assert crd.ndim == 1
            if crd[p] == lvls[level]:
                return self._locate(formats, lvls, level + 1, p)
        elif fmt == LevelFormat.DELTA:
            assert is_unique(prop)
            pos = self.pos(level).tensor
            crd = self.crd(level).tensor
            assert pos.ndim == 1 and crd.ndim == 1
            corig = 0
            lo = pos[p].item()
            hi = pos[p + 1].item()
            for i in range(lo, hi):
                corig += crd[i].item()
                if corig == lvls[level]:
                    return self._locate(formats, lvls, level + 1, i)
                corig += 1
        else:
            raise AssertionError(f"Unsupported: {fmt}")
        return None

    @classmethod
    def _from_dense(cls, operand):
        # Wrap operand to canonicalize attributes.
        operand = tensor_wrapper.wrap_operand(operand)

        if not is_contiguous_and_dense(operand.shape, operand.strides):
            raise NotImplementedError("A UST cannot be currently created from strided tensors with holes (that is, not dense).")

        # Axis order, to determine UST dense format.
        axis_order = _axis_order_in_memory(operand.shape, operand.strides)
        num_dimensions = len(axis_order)
        dimensions = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(num_dimensions)]
        levels = {dimensions[d]: LevelFormat.DENSE for d in reversed(axis_order)}

        # UST format.
        if num_dimensions == 0:
            name = "Scalar"
        elif num_dimensions == 1:
            name = "DenseVector"
        elif num_dimensions == 2:
            name = "Densed" + ("Left" if axis_order == (0, 1) else "Right")
        else:
            name = f"Dense{num_dimensions}D-" + "-".join(str(d) for d in reversed(axis_order))
        tensor_format = TensorFormat(dimensions, levels, name=name)

        # UST data.
        index_type = _get_smallest_index_type(max(operand.shape) if num_dimensions > 0 else 0)
        values = operand.memory_buffer()
        dtype = values.dtype

        # Create UST.
        ust = cls(operand.shape, tensor_format=tensor_format, index_type=index_type, dtype=dtype)
        ust._val = values

        ust.wrapped_operand = operand

        return ust
