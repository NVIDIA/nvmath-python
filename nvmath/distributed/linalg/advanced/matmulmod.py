# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["MatmulComputeType", "Matmul", "matmul"]

import logging
import typing
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
import numpy as np

import nvmath.distributed
from nvmath import memory
from nvmath._internal.layout import is_contiguous_and_dense
from nvmath._utils import CudaDataType
from nvmath.bindings import (
    cublas,
    cublasMp,  # type: ignore
)
from nvmath.distributed._internal import tensor_wrapper
from nvmath.distributed._internal.tensor_ifc import DistributedTensor
from nvmath.distributed.distribution import BlockCyclic, BlockNonCyclic, Distribution, ProcessGrid
from nvmath.distributed.linalg._internal import matmul_desc_ifc
from nvmath.distributed.linalg._internal.epilog_protocol import (
    EPILOG_INPUT_HANDLERS_MAP,
    EPILOG_MINIMUM_VERSIONS_MAP,
    EPILOG_OUTPUT_HANDLERS_MAP,
    EpilogOutputHandler,
)
from nvmath.distributed.linalg.advanced import MatmulEpilog, _configuration
from nvmath.internal import formatters, typemaps, utils
from nvmath.linalg._internal.typemaps import (
    COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE,
    NAMES_TO_DEFAULT_COMPUTE_TYPE,
    NAMES_TO_DEFAULT_SCALE_TYPE,
    SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE,
    SUPPORTED_TYPES,
)
from nvmath.linalg._internal.utils import (
    calculate_strides,
)

from ._configuration import MatmulOptions, matrix_qualifiers_dtype

MatmulComputeType = cublas.ComputeType

EpilogInputTraits = namedtuple("EpilogInputTraits", ["dtype", "extents", "strides"])


@dataclass
class MatrixLayout:
    """An internal data class for capturing the local tensor layout."""

    shape: Sequence[int]
    strides: Sequence[int]
    is_transpose: bool = False
    is_conjugate: bool = False  # Used to support is_conjugate via conjugate_transpose.


@dataclass
class MMTraits:
    """An internal data class for capturing the matrix multiplication traits. The
    result traits are captured separately, because we need to wait for the
    epilog to be provided.
    """

    M: int  # global matrix size M
    N: int  # global matrix size N
    K: int  # global matrix size K
    a_layout: MatrixLayout  # local
    b_layout: MatrixLayout  # local
    c_layout: MatrixLayout | None  # local
    d_layout: MatrixLayout | None  # local
    # NOTE: cuBLASMp doesn't support batched matmul so batch size is currently always 0.
    batch_count: int = 0
    batch_shape: Sequence[int] = field(default_factory=list)
    batch_axis_order: Sequence[int] = field(default_factory=list)


@dataclass
class _ProblemSpec:
    """This is used in a custom reduction to check that the Matmul problem specification
    is consistent across processes, and to infer global information (e.g. global shape
    of distributed matrices)."""

    @dataclass
    class Options:
        """
        This is used for _ProblemSpec instead of MatmulOptions because it's going
        to be serialized as part of the custom reduction of the _ProblemSpec, and
        we want to control which fields are included (for example we don't need
        the logger).
        """

        def __init__(self, options: MatmulOptions):
            self.inplace = options.inplace
            self.compute_type = options.compute_type
            self.scale_type = options.scale_type
            self.result_type = options.result_type
            self.result_amax = options.result_amax
            self.block_scaling = options.block_scaling
            self.blocking = options.blocking

        inplace: bool = False
        compute_type: int | None = None
        scale_type: int | None = None
        result_type: int | None = None
        result_amax: bool = False
        block_scaling: bool = False
        blocking: Literal[True, "auto"] = "auto"

    shapes: list[list[int]]  # shape of each operand
    is_F: list[bool]  # Is F memory layout (of each operand)
    operand_dtypes: list[str]  # dtype of each operand
    packages: list[Literal["numpy", "cupy", "torch"]]  # package of each operand
    memory_spaces: list[Literal["cuda", "cpu"]]  # memory space of each operand
    distributions: list[Distribution]  # distribution of A, B and C/D
    options: Options  # Matmul options
    device_ids: list[int | Literal["cpu"]]  # device_id of each operand
    compute_capability: tuple[int, ...]  # compute capability of the execution space device
    alpha: float
    beta: float
    qualifiers: np.ndarray
    nranks: int
    rank: int  # only valid if is_leaf=True
    lib_version: int

    # is_leaf=True means that this is the _ProblemSpec of a process before reducing
    # with that of another process.
    is_leaf: bool = True


def _problem_spec_reducer(p1: _ProblemSpec, p2: _ProblemSpec):
    try:
        if isinstance(p1, Exception):
            return p1  # propagate exception

        if isinstance(p2, Exception):
            return p2  # propagate exception

        if not (p1.lib_version == p2.lib_version >= 800):
            return ValueError("cublasMp >= 0.8.0 required")

        num_operands = len(p1.operand_dtypes)
        if num_operands != len(p2.operand_dtypes):
            return ValueError("The number of operands doesn't match across processes")

        if num_operands not in (2, 3):
            return ValueError("The number of operands must be 2 or 3")

        def check_dtype(dtype, operand_name: str):
            if dtype not in SUPPORTED_TYPES:
                raise ValueError(f"The dtype of operand {operand_name} ({dtype}) is not supported.")

        operand_name = "ABC"

        for i in range(num_operands):
            if p1.operand_dtypes[i] != p2.operand_dtypes[i]:
                return ValueError(
                    f"Operand {operand_name[i]} dtype does not match across processes: "
                    f"{p1.operand_dtypes[i]} != {p2.operand_dtypes[i]}"
                )
            check_dtype(p1.operand_dtypes[i], operand_name[i])

        def _check_extents(shape: list[int], name: str):
            if len(shape) > 2:
                raise ValueError("Batched matmul is not supported")
            if name == "C" and len(shape) != 2:
                raise ValueError(
                    "In order to avoid broadcasting behavior ambiguity, `c` must be 2-D. "
                    "Use a singleton dimension to convert your input array to 2-D."
                )
            # TODO: allow broadcasting A and B if 1D.
            if len(shape) != 2:
                raise ValueError("Operands must be two-dimensional")
            if any(e <= 0 for e in shape):
                message = (
                    f"The specified extents {shape} for operand {name} are not valid. The extents must be strictly positive. "
                )
                raise ValueError(message)

        for p in (p1, p2):
            if p.is_leaf:
                for i in range(num_operands):
                    _check_extents(p.shapes[i], operand_name[i])

                if len(set(p.packages)) != 1:
                    return ValueError(
                        f"The operands on process {p.rank} don't belong to the same package: got operand packages {p.packages}"
                    )

                if len(set(p.memory_spaces)) != 1:
                    return ValueError(
                        f"The operands on process {p.rank} are not in the same memory space: got "
                        f"operand memory spaces {p.memory_spaces}"
                    )

                if len(set(p.device_ids)) != 1:
                    return ValueError(
                        f"The operands on process {p.rank} are not on the same device: got operand device IDs {p.device_ids}"
                    )

                input_type_width = typemaps.NAME_TO_DATA_WIDTH[p.operand_dtypes[0]]
                if input_type_width <= 8:
                    if p.compute_capability < (8, 9):
                        return RuntimeError(
                            "FP8 requires a device with compute capability 8.9 or higher "
                            "(Ada, Hopper, Blackwell or newer architecture). The compute "
                            f"capability of process {p.rank}'s device is {p.compute_capability}"
                        )
                    if p.options.block_scaling and p.compute_capability < (10, 0):
                        return RuntimeError(
                            "MXFP8 requires a device with compute capability 10.0 or higher "
                            "(Blackwell or newer architecture). The compute capability of "
                            f"process {p.rank}'s device is {p.compute_capability}"
                        )

                p.qualifiers = p.qualifiers if p.qualifiers is not None else np.zeros((3,), dtype=matrix_qualifiers_dtype)
                if p.qualifiers.dtype != matrix_qualifiers_dtype:
                    return ValueError(
                        "The qualifiers must be specified as a NumPy array of length 3 "
                        "corresponding to the operands A, B, and C of type "
                        "'matrix_qualifiers_dtype'."
                    )

        for i in range(num_operands):
            if len(p1.shapes[i]) != len(p2.shapes[i]):
                return ValueError(f"The number of dimensions of the operand {operand_name[i]} is inconsistent across processes")

        if p1.packages[0] != p2.packages[0]:
            return ValueError("operands don't belong to the same package on all processes")

        if p1.memory_spaces[0] != p2.memory_spaces[0]:
            return ValueError('operands are not in the same memory space ("cpu", "cuda") on all processes')

        if p1.options != p2.options:
            return ValueError(f"options are inconsistent across processes: {p1.options} != {p2.options}")

        if p1.alpha != p2.alpha:
            return ValueError(f"alpha does not match across processes: {p1.alpha} != {p2.alpha}")

        if p1.beta != p2.beta:
            return ValueError(f"beta does not match across processes: {p1.beta} != {p2.beta}")

        if not np.array_equal(p1.qualifiers, p2.qualifiers):
            return ValueError("The qualifiers don't match across processes")

        if num_operands == 2 and p1.options.inplace:
            return ValueError("The operation cannot be inplace if operand C is not provided.")

        if len(p1.distributions) != 3 or len(p2.distributions) != 3:
            return ValueError("Must provide distributions for A, B and C/D")

        # Check that distribution of operands is the same on every process.
        for i, (d1, d2) in enumerate(zip(p1.distributions, p2.distributions, strict=False)):
            if d1 != d2:
                return ValueError(f"Distribution for {operand_name[i]} doesn't match across processes: {d1} != {d2}")

        for p in (p1, p2):
            if p.is_leaf:
                p.distributions = [d.to(BlockCyclic, ndim=2, copy=True) for d in p.distributions]
                for i, d in enumerate(p.distributions):
                    assert isinstance(d, BlockCyclic)  # only for type checker
                    if i == num_operands:
                        break
                    # To calculate the global shape when using 2D block distribution, we
                    # ignore the rows of processes that aren't in column 0 of the process
                    # grid, and the columns of processes that aren't in row 0 of the process
                    # grid (by setting the rows/columns to 0). We could do the same for 1D,
                    # but by preserving the shape info for 1D we can do some extra checks
                    # below.
                    if not d.process_grid._is_1d_distribution():
                        nprow, npcol = d.process_grid.shape
                        myprow = p.rank % nprow if d.process_grid.layout == ProcessGrid.Layout.COL_MAJOR else p.rank // npcol
                        mypcol = p.rank // nprow if d.process_grid.layout == ProcessGrid.Layout.COL_MAJOR else p.rank % npcol
                        if myprow != 0:
                            p.shapes[i][1] = 0
                        if mypcol != 0:
                            p.shapes[i][0] = 0

        # Determine the memory layout shared by all processes.
        for i in range(num_operands):
            p1.is_F[i] &= p2.is_F[i]
            if not p1.is_F[i]:
                return ValueError(f"Operand {operand_name[i]} doesn't have column-major (Fortran) memory layout")

        # Calculate global shape based on process grid.
        for i in range(num_operands):
            p_grid = cast(BlockCyclic, p1.distributions[i]).process_grid
            partitioned_dims = (0,) if p_grid._is_row_wise() else (1,) if p_grid._is_col_wise() else (0, 1)

            if len(partitioned_dims) == 1 and any(
                p1.shapes[i][j] != p2.shapes[i][j] for j in (0, 1) if j != partitioned_dims[0]
            ):
                return ValueError(
                    "The problem size is inconsistent across processes:" + str(p1.shapes) + " vs " + str(p2.shapes)
                )

            if p1 is not p2:  # with nranks==1 p1 is p2
                # Reduce the partitioned dimensions to get the global size.
                for dim in partitioned_dims:
                    p1.shapes[i][dim] += p2.shapes[i][dim]

    except Exception as e:
        return e
    p1.is_leaf = False
    return p1


SHARED_MM_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_MM_DOCUMENTATION.update(
    {
        "a": """\
A distributed tensor representing the first operand to the matrix multiplication (see `Semantics`_).
The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and
:class:`torch.Tensor`.""".replace("\n", " "),
        #
        "b": """\
A distributed tensor representing the second operand to the matrix multiplication (see `Semantics`_).
The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and
:class:`torch.Tensor`.""".replace("\n", " "),
        #
        "c": """\
(Optional) A distributed tensor representing the operand to add to the matrix multiplication
result (see `Semantics`_). The currently supported types are :class:`numpy.ndarray`,
:class:`cupy.ndarray`, and :class:`torch.Tensor`.""".replace("\n", " "),
        #
        "distributions": """\
Sequence specifying the distribution across processes of matrices A, B and C/D. The distribution needs to
be BlockCyclic or compatible.""".replace("\n", " "),
        #
        "alpha": """\
The scale factor for the matrix multiplication term as a real or complex number. The default is
:math:`1.0`.""".replace("\n", " "),
        #
        "beta": """\
The scale factor for the matrix addition term as a real or complex number. A value for `beta` must be provided if
operand `c` is specified.""".replace("\n", " "),
        #
        "quantization_scales": """\
Specify scale factors for the matrix multiplication as a :class:`~nvmath.distributed.linalg.advanced.MatmulQuantizationScales`
object. Alternatively, a `dict` containing the parameters for the
:class:`~nvmath.distributed.linalg.advanced.MatmulQuantizationScales`
constructor can also be provided. The scale factors can be provided as scalars or tensors.
If a scale factor is provided as a tensor, it must be from the same package and on the same
memory space (CPU or GPU device) as the operands of the matmul.
If a scale factor is provided as a scalar, and the execution space is GPU,
a CPU->GPU copy is inevitable. To avoid this copy, provide
the quantization scale as one-element array on the GPU.
Allowed and required only for narrow-precision (FP8 and lower) operations.""".replace("\n", " "),
        #
        "epilog": """\
Specify an epilog :math:`F` as an object of type :class:`MatmulEpilog` to apply to the result of the matrix
multiplication: :math:`F(\\alpha A @ B + \\beta C`). The default is no epilog. See `cuBLASMp documentation
<https://docs.nvidia.com/cuda/cublasmp/usage/types.html#cublasmpmatmulepilogue-t>`_ for the list of
available epilogs.""".replace("\n", " "),
        #
        "epilog_inputs": """\
Specify the additional inputs needed for the selected epilog as a dictionary, where the key is the epilog input name and
the value is the epilog input. The epilog input must be a tensor with the same package and in the same memory space as
the operands (see the constructor for more information on the operands). If the required epilog inputs are not provided,
an exception is raised that lists the required epilog inputs. Some epilog inputs are generated by other epilogs. For
example, the epilog input for :class:`MatmulEpilog.DRELU` is generated by matrix multiplication with the same operands
using :class:`MatmulEpilog.RELU_AUX`. """.replace("\n", " "),
        #
        "qualifiers": """\
Specify the matrix qualifiers as a :class:`numpy.ndarray` of
:class:`~nvmath.distributed.linalg.advanced.matrix_qualifiers_dtype` objects of length 3
corresponding to the operands `a`, `b`, and `c`. See
:ref:`matrix-tensor-qualifiers` for the motivation behind qualifiers.""".replace("\n", " "),
        #
        "options": """\
Specify options for the matrix multiplication as a
:class:`~nvmath.distributed.linalg.advanced.MatmulOptions` object. Alternatively, a `dict` containing
the parameters for the ``MatmulOptions`` constructor can also be provided. If not specified, the
value will be set to the default-constructed ``MatmulOptions`` object.""".replace("\n", " "),
        #
        "preferences": """\
This parameter specifies the preferences for planning as a :class:`MatmulPlanPreferences` object. Alternatively, a
dictionary containing the parameters for the :class:`MatmulPlanPreferences` constructor can also be provided. If not
specified, the value will be set to the default-constructed :class:`MatmulPlanPreferences` object.
""".replace("\n", " "),
        #
        "result": """\
The result of the specified matrix multiplication (epilog applied), which remains on the same device and belongs to the
same package as the input operands. If an epilog (like :attr:`nvmath.distributed.linalg.advanced.MatmulEpilog.RELU_AUX`) that
results in extra output is used, or an extra output is requested (for example by setting
:attr:`~nvmath.distributed.linalg.advanced.MatmulOptions.result_amax` option in ``options`` argument),
a tuple is returned with the first element being the matrix multiplication result (epilog applied) and the second element
being the auxiliary output provided as a `dict`. """.replace("\n", " "),
        #
        "narrow_precision": """\
        Matrix multiplication with narrow-precision operands is supported, in both FP8 and MXFP8 formats.

        .. note::

            **FP8 requires a device with compute capability 8.9 or higher** (Ada, Hopper, Blackwell or newer architecture).
            **MXFP8 requires a device with compute capability 10.0 or higher** (Blackwell or newer architecture).
            Please refer to the `compute capability table <https://developer.nvidia.com/cuda-gpus>`_
            to check the compute capability of your device.

        For FP8 operations:

        * For each operand a scaling factor needs to be specified via ``quantization_scales`` argument.
        * Maximum absolute value of the result (amax) can be requested via
          :attr:`~nvmath.distributed.linalg.advanced.MatmulOptions.result_amax` option in ``options`` argument.
        * Custom result type (both FP8 and non-FP8) can be requested via
          :attr:`~nvmath.distributed.linalg.advanced.MatmulOptions.result_type` option in ``options`` argument.

        For MXFP8 operations:

        * To enable MXFP8 operations, :attr:`~nvmath.distributed.linalg.advanced.MatmulOptions.block_scaling` option
          must be set to ``True``.
        * Block scaling factors need to be specified via ``quantization_scales`` argument.
        * Utilities in :mod:`nvmath.distributed.linalg.advanced.helpers.matmul` can be used to create and modify
          block scaling factors.
        * When MXFP8 is used and the result type is a narrow-precision data type, the auxiliary output
          ``"d_out_scale"`` will be returned in the auxiliary output tensor. It will contain the scales
          that were used for the result quantization.

        Please refer to the examples and narrow-precision operations tutorial for more details.
        cuBLASMp follows cuBLAS specification and usage for FP8 and MXFP8 formats, scaling modes,
        scaling factor layouts, etc. For more details see the
        `cublasLtMatmul documentation <https://docs.nvidia.com/cuda/cublas/#cublasltmatmul>`_.
""".strip(),
        #
        "semantics": """\
        .. _semantics:

        The semantics of the matrix multiplication follows :external:py:data:`numpy.matmul` semantics, with some restrictions on
        broadcasting. In addition, the semantics for the fused matrix addition are described below:

        * For in-place matrix multiplication (where the result is written into `c`) the result has the same shape as `c`.
""".strip(),
    }
)


class InvalidMatmulState(Exception):
    pass


@utils.docstring_decorator(SHARED_MM_DOCUMENTATION, skip_missing=False)
class Matmul:
    """
    Create a stateful object encapsulating the specified distributed matrix multiplication
    computation :math:`\\alpha a @ b + \\beta c` and the required resources to perform the
    operation.  A stateful object can be used to amortize the cost of preparation (planning
    in the case of matrix multiplication) across multiple executions (also see the
    :ref:`Stateful APIs <host api types>` section).

    The function-form API :func:`matmul` is a convenient alternative to using stateful
    objects for *single* use (the user needs to perform just one matrix multiplication, for
    example), in which case there is no possibility of amortizing preparatory costs. The
    function-form APIs are just convenience wrappers around the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific matrix multiplication operation.
    3. **Execution**: Perform the matrix multiplication computation with :meth:`execute`.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on what's happening in the various phases described above can be
    obtained by passing in a :class:`logging.Logger` object to :class:`MatmulOptions` or by
    setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        distributions: {distributions}

        alpha: {alpha}

        beta: {beta}

        qualifiers: {qualifiers}

        quantization_scales: {quantization_scales}

        options: {options}

        stream: {stream}

    Semantics:
        {semantics}

    Narrow-precision support:
        {narrow_precision}

    .. seealso::
        :meth:`plan`, :meth:`reset_operands`, :meth:`execute`

    Examples:

        >>> import numpy as np
        >>> import nvmath.distributed
        >>> from nvmath.distributed.distribution import Slab
        >>> from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

        Get process group used to initialize nvmath.distributed (for information on
        initializing ``nvmath.distributed``, you can refer to the documentation or to the
        Matmul examples in `nvmath/examples/distributed/linalg/advanced
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_):

        >>> process_group = nvmath.distributed.get_context().process_group

        Get my process rank:

        >>> rank = process_group.rank

        Create two 2-D float64 ndarrays on the CPU (using Slab distributions to distribute
        the matrices across processes):

        >>> M, N, K = 1024, 1024, 1024
        >>> a_shape = Slab.X.shape(rank, (K, M))
        >>> b_shape = Slab.X.shape(rank, (K, N))
        >>> a = np.asfortranarray(np.random.rand(*a_shape))
        >>> b = np.asfortranarray(np.random.rand(*b_shape))

        We will define a matrix multiplication operation followed by an AllReduce epilog
        using the specialized matrix multiplication interface.

        Create a Matmul object encapsulating the problem specification above:

        >>> qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
        >>> qualifiers[0]["is_transpose"] = True  # a is transposed
        >>> distributions = [Slab.X, Slab.X, Slab.Y]
        >>> mm = nvmath.distributed.linalg.advanced.Matmul(
        ...     a, b, distributions=distributions, qualifiers=qualifiers
        ... )

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`MatmulOptions`).

        Next, plan the operation. The epilog is specified, and optionally, preferences can
        be specified for planning:

        >>> epilog = nvmath.distributed.linalg.advanced.MatmulEpilog.ALLREDUCE
        >>> mm.plan(epilog=epilog)

        Now execute the matrix multiplication, and obtain the result `r1` as a NumPy
        ndarray.

        >>> r1 = mm.execute()

        Finally, free the object's resources. To avoid having to explicitly make this
        call, it's recommended to use the Matmul object as a context manager as shown below,
        if possible.

        >>> mm.free()

        Note that all :class:`Matmul` methods execute on the current stream by default.
        Alternatively, the `stream` argument can be used to run a method on a specified
        stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        >>> device_id = nvmath.distributed.get_context().device_id
        >>> import cupy as cp
        >>> with cp.cuda.Device(device_id):
        ...     a = cp.asfortranarray(cp.random.rand(*a_shape))
        ...     b = cp.asfortranarray(cp.random.rand(*b_shape))

        Create a Matmul object encapsulating the problem specification described earlier
        and use it as a context manager.

        >>> with nvmath.distributed.linalg.advanced.Matmul(
        ...     a, b, distributions=distributions, qualifiers=qualifiers
        ... ) as mm:
        ...     mm.plan(epilog=epilog)
        ...
        ...     # Execute the operation to get the first result.
        ...     r1 = mm.execute()
        ...
        ...     # Update operands A and B in-place (see reset_operands() for an
        ...     # alternative).
        ...     with cp.cuda.Device(device_id):
        ...         a[:] = cp.random.rand(*a_shape)
        ...         b[:] = cp.random.rand(*b_shape)
        ...
        ...     # Execute the operation to get the new result.
        ...     r2 = mm.execute()


        All the resources used by the object are released at the end of the block.

        Further examples can be found in the
        `nvmath/examples/distributed/linalg/advanced/matmul
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_
        directory.
    """

    def __init__(
        self,
        a,
        b,
        /,
        c=None,
        *,
        distributions: Sequence[Distribution],
        alpha=None,
        beta=None,
        qualifiers=None,
        quantization_scales=None,
        options=None,
        stream: utils.AnyStream | int | None = None,
    ):
        distributed_ctx = nvmath.distributed.get_context()
        if distributed_ctx is None:
            raise RuntimeError(
                "nvmath.distributed has not been initialized. Refer to "
                "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/runtime.html"
                " for more information."
            )
        if distributed_ctx.nccl_comm is None:
            raise RuntimeError("nvmath.distributed wasn't initialized with NCCL backend")
        self.process_group = process_group = distributed_ctx.process_group
        self.rank = rank = process_group.rank
        self.nranks = nranks = process_group.nranks

        self.options = options = cast(
            MatmulOptions, utils.check_or_create_options(MatmulOptions, options, "Distributed matrix multiplication options")
        )
        self.logger = options.logger if options.logger is not None else logging.getLogger()

        # The matrix multiplication has two required operands 'a' and 'b', and one optional
        # operand 'c'.
        a = tensor_wrapper.wrap_operand(a)
        b = tensor_wrapper.wrap_operand(b)
        if c is not None:
            c = tensor_wrapper.wrap_operand(c)

        operands = [a, b, c] if c is not None else [a, b]

        version = cublasMp.get_version()

        problem_spec = _ProblemSpec(
            distributions=list(distributions),
            shapes=[list(o.shape) for o in operands],  # local shapes
            operand_dtypes=[o.dtype for o in operands],
            options=_ProblemSpec.Options(options),
            packages=[o.name for o in operands],
            memory_spaces=[o.device for o in operands],
            device_ids=[o.device_id for o in operands],
            compute_capability=tuple(Device(distributed_ctx.device_id).compute_capability),
            alpha=alpha,
            beta=beta,
            qualifiers=qualifiers,
            is_F=[sorted(o.strides) == list(o.strides) and is_contiguous_and_dense(o.shape, o.strides) for o in operands],
            lib_version=version,
            nranks=nranks,
            rank=rank,
        )

        if nranks > 1:
            problem_spec = process_group.allreduce_object(problem_spec, op=_problem_spec_reducer)
        else:
            # Ensure we error-check with one rank.
            problem_spec = _problem_spec_reducer(problem_spec, problem_spec)
        if isinstance(problem_spec, Exception):
            # There is an error or inconsistency in the problem spec across processes.
            # Note that since this comes from an allreduce, all processes will have
            # received the same exception.
            raise problem_spec

        self.distributions = distributions = cast(Sequence[BlockCyclic], problem_spec.distributions)

        if options.inplace and options.result_type is not None:
            self.logger.warning(
                f"Matmul: The provided result type {options.result_type.name} in options is ignored since \
the operation is in-place."
            )
        self.inplace = options.inplace

        self.logger.info("= SPECIFICATION PHASE =")
        if self.inplace:
            self.logger.info("The MM operation will be performed in-place (the result will be written into operand C).")
        self.logger.info("For performance and debugging hints, use CUBLASMP_LOG_LEVEL=5 and CUBLASLT_LOG_LEVEL=5")
        self.logger.info(f"The data type of operand A is '{a.dtype}', and that of operand B is '{b.dtype}'.")

        self.num_operands = len(operands)
        if c is not None:
            self.logger.info(f"The data type of operand C is {c.dtype}.")
            if beta is None:
                raise ValueError("A value for beta must be provided if operand C is provided.")

        if (a.dtype, b.dtype) not in NAMES_TO_DEFAULT_SCALE_TYPE:
            raise ValueError(f"Unsupported combination of dtypes for operands A {a.dtype} and B {b.dtype}.")

        operand_name = "ABC"
        for i in range(self.num_operands):
            global_shape = tuple(problem_spec.shapes[i])
            self.logger.info(f"The global shape of operand {operand_name[i]} is {global_shape}.")

        self.logger.info(f"The distribution of operand A is {self.distributions[0]}")
        self.logger.info(f"The distribution of operand B is {self.distributions[1]}")
        self.logger.info(f"The distribution of operand C/D is {self.distributions[2]}")

        # Currently, a.dtype != b.dtype is only supported for FP8 (different FP8 kinds are
        # allowed), so we assume that A and B have equal width.
        self.input_type_width = typemaps.NAME_TO_DATA_WIDTH[a.dtype]

        assert self.num_operands == 2 or self.num_operands == 3, "Internal Error."

        # Infer the library package & device ID the operands belong to.
        # Note that wrappers are kept even after release_operands();
        # see that method's docstring for details.
        self.operands: list[DistributedTensor] = operands

        self.package = utils.get_operands_package(operands)
        self.memory_space = "cuda"
        self.device_id = utils.get_operands_device_id(operands)
        if self.device_id == "cpu":
            if self.package == "numpy":
                self.package = "cuda"
            self.memory_space = "cpu"
            self.device_id = distributed_ctx.device_id
        elif self.device_id != distributed_ctx.device_id:
            raise RuntimeError(
                "The operands are not on the same device as the one assigned to the distributed "
                f"runtime on this process: operands' device ID is {self.device_id} and the runtime "
                f"device ID is {distributed_ctx.device_id}"
            )
        self.logger.info(
            f"The input operands' memory space is {self.memory_space}, and the execution space is on device {self.device_id}."
        )

        nccl_comm = distributed_ctx.nccl_comm

        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for the Matmul ctor is {stream_holder.obj}.")

        # Copy operands to device (and store reference to CPU operand), if needed.
        self.cpu_c_ref = None
        if self.memory_space == "cpu":
            if self.inplace:
                self.cpu_c_ref = self.operands[2]  # Hold reference, needed for inplace operations.
            # Some of the comm overlap algorithms in cuBLASMp will perform better
            # when some of the operands are already on symmetric memory (e.g. AG+GEMM
            # when B is on symmetric memory).
            self.operands = [o.to(self.device_id, stream_holder) for o in self.operands]

        self._set_result_sheap_flag()

        # Set qualifiers.
        self.qualifiers = problem_spec.qualifiers
        # Set qualifiers based on torch lazy conjugation flag if not provided.
        self.qualifiers[0]["is_conjugate"] = self.qualifiers[0]["is_conjugate"] ^ self.operands[0].is_conjugate
        self.qualifiers[1]["is_conjugate"] = self.qualifiers[1]["is_conjugate"] ^ self.operands[1].is_conjugate
        self.lazy_conjugation = (self.operands[0].is_conjugate, self.operands[1].is_conjugate, False)
        if self.num_operands == 3:
            self.qualifiers[2]["is_conjugate"] = self.qualifiers[2]["is_conjugate"] ^ self.operands[2].is_conjugate
            if self.qualifiers[2]["is_conjugate"]:
                raise ValueError("The conjugate flag is currently not supported for operand C.")
            if self.qualifiers[2]["is_transpose"]:
                raise ValueError("The transpose flag is currently not supported for operand C.")
        for i in range(2):
            if self.qualifiers[i]["is_conjugate"] and not self.qualifiers[i]["is_transpose"]:
                raise ValueError("Conjugate is not supported without transpose")

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        # The result class is that of the first wrapped device operand.
        self.result_class = self.operands[0].__class__

        # Create cuBLASMp handle.
        with utils.device_ctx(self.device_id):
            self.handle: int = cublasMp.create(stream_holder.ptr)

        # Determine the data types for a and b.
        self.a_dtype = typemaps.NAME_TO_DATA_TYPE[a.dtype]
        self.b_dtype = typemaps.NAME_TO_DATA_TYPE[b.dtype]
        self.a_dtype_name = a.dtype
        self.b_dtype_name = b.dtype

        self.is_complex = "complex" in self.a_dtype_name or "complex" in self.b_dtype_name

        for i, dtype_name in enumerate((a.dtype, b.dtype)):
            if self.qualifiers[i]["is_conjugate"] and "complex" not in dtype_name:
                raise ValueError("The conjugate flag only applies to complex operands")

        # Determine the data types for c and d.
        self.d_dtype = None if self.inplace else options.result_type
        if self.num_operands == 3:
            self.c_dtype = typemaps.NAME_TO_DATA_TYPE[c.dtype]
            if self.d_dtype is None:
                self.d_dtype = self.c_dtype
        elif self.num_operands == 2:
            if self.d_dtype is None:
                self.d_dtype = self.a_dtype
            if self.d_dtype in (CudaDataType.CUDA_R_8F_E5M2, CudaDataType.CUDA_R_8F_E4M3):
                self.c_dtype = CudaDataType.CUDA_R_16F
            else:
                self.c_dtype = self.d_dtype
        self.c_dtype_name = typemaps.DATA_TYPE_TO_NAME[self.c_dtype]
        self.d_dtype_name = typemaps.DATA_TYPE_TO_NAME[self.d_dtype]
        self.c_dtype_width = typemaps.NAME_TO_DATA_WIDTH[self.c_dtype_name]
        self.d_dtype_width = typemaps.NAME_TO_DATA_WIDTH[self.d_dtype_name]

        self.logger.info(f"The data type for the result D is '{self.d_dtype_name}'.")

        def assert_valid_compute_type(compute_type):
            if compute_type not in COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE["real"]:
                message = f"Unsupported compute type. The compute type '{repr(compute_type)}' is currently not supported."
                raise ValueError(message)

        # Determine the scale type.
        if options.scale_type is None:
            if options.compute_type is not None:
                assert_valid_compute_type(options.compute_type)
                if self.is_complex:
                    scale_type_map = COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE["complex"]
                else:
                    scale_type_map = COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE["real"]
                self.scale_type = scale_type_map[options.compute_type]
            else:
                self.scale_type = NAMES_TO_DEFAULT_SCALE_TYPE[(self.a_dtype_name, self.b_dtype_name)]
            self.scale_type_name = typemaps.DATA_TYPE_TO_NAME[self.scale_type]
        else:
            self.scale_type = options.scale_type
            if self.scale_type not in SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE:
                message = f"Unsupported scale type. The data type '{repr(self.scale_type)}' is currently not supported."
                raise ValueError(message)
            self.scale_type_name = typemaps.DATA_TYPE_TO_NAME[self.scale_type]
        self.logger.info(f"The scale type is '{self.scale_type_name}'.")

        # Determine the compute type.
        if options.compute_type is None:
            if options.scale_type is not None:
                self.compute_type = SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE[options.scale_type]
            else:
                self.compute_type = NAMES_TO_DEFAULT_COMPUTE_TYPE[(self.a_dtype_name, self.b_dtype_name)]
        else:
            self.compute_type = options.compute_type
        assert_valid_compute_type(self.compute_type)
        self.logger.info(f"The compute type is {self.compute_type.name}.")

        def is_supported(atype, btype, compute_type, scale_type):
            ct = cublas.ComputeType
            st = CudaDataType
            abtype = atype if atype == btype else (atype, btype)
            if compute_type in (ct.COMPUTE_16F, ct.COMPUTE_16F_PEDANTIC):
                return scale_type == st.CUDA_R_16F and abtype == "float16"
            elif compute_type == ct.COMPUTE_32F_PEDANTIC:
                if scale_type == st.CUDA_R_32F:
                    return abtype in ("float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2")
                elif scale_type == st.CUDA_C_32F:
                    return abtype == "complex64"
            elif compute_type == ct.COMPUTE_32F:
                if scale_type == st.CUDA_R_32F:
                    return abtype in (
                        "float32",
                        "bfloat16",
                        "float16",
                        "float8_e4m3fn",
                        "float8_e5m2",
                        ("float8_e4m3fn", "float8_e5m2"),
                        ("float8_e5m2", "float8_e4m3fn"),
                    )
                elif scale_type == st.CUDA_C_32F:
                    return abtype == "complex64"
            elif compute_type in (
                ct.COMPUTE_32F_FAST_16F,
                ct.COMPUTE_32F_FAST_16BF,
                ct.COMPUTE_32F_FAST_TF32,
                ct.COMPUTE_32F_EMULATED_16BFX9,
            ):
                if scale_type == st.CUDA_R_32F:
                    return abtype == "float32"
                if scale_type == st.CUDA_C_32F:
                    return abtype == "complex64"
            elif compute_type in (ct.COMPUTE_64F, ct.COMPUTE_64F_PEDANTIC, ct.COMPUTE_64F_EMULATED_FIXEDPOINT):
                if scale_type == st.CUDA_R_64F:
                    return abtype == "float64"
                if scale_type == st.CUDA_C_64F:
                    return abtype == "complex128"
            return False

        if not is_supported(self.a_dtype_name, self.b_dtype_name, self.compute_type, self.scale_type):
            raise ValueError(
                f"Selected scale_type={repr(self.scale_type)} compute_type={repr(self.compute_type)} "
                + f"are not supported for data types {self.a_dtype_name} (A) and {self.b_dtype_name} (B)."
            )

        # Set alpha and beta.
        self.alpha = np.zeros((1,), dtype=self.scale_type_name)
        try:
            self.alpha[0] = alpha if alpha is not None else 1
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'.") from e

        self.beta = np.zeros((1,), dtype=self.scale_type_name)
        if beta is not None and self.num_operands == 2:
            self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
        try:
            self.beta[0] = beta if beta is not None and self.num_operands == 3 else 0
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'.") from e

        # Set narrow-precision (FP8 and lower) quantization_scales.
        if self.input_type_width <= 8:
            self.quantization_scales = self._validate_operand_scales(quantization_scales, all_required=True)
        elif quantization_scales is not None:
            self.logger.warning(
                "Matmul: The provided scales are ignored, since they are only applicable to narrow-precision (FP8 and lower) "
                "operations."
            )

        if self.options.result_amax and self.d_dtype_width > 8:
            raise ValueError("result_amax=True is allowed only for narrow-precision (FP8 and lower) results")

        # Check operands alignment if needed
        if self.input_type_width <= 8:
            for operand, operand_name in zip(self.operands, "ABC", strict=False):
                if operand.data_ptr % 16 != 0:
                    raise ValueError(
                        f"For narrow-precision (FP8 and lower) multiplication, operand {operand_name} should be aligned to 16 "
                        "bytes."
                    )

        # Capture operand extents and strides for consistency check when resetting operands.
        self.operand_extents = tuple(o.shape for o in self.operands)
        self.operand_strides = tuple(o.strides for o in self.operands)

        # Create operand layouts.
        a_layout = MatrixLayout(
            shape=self.operands[0].shape,
            strides=self.operands[0].strides,
            is_transpose=bool(self.qualifiers[0]["is_transpose"]),
            is_conjugate=bool(self.qualifiers[0]["is_conjugate"]),
        )
        b_layout = MatrixLayout(
            shape=self.operands[1].shape,
            strides=self.operands[1].strides,
            is_transpose=bool(self.qualifiers[1]["is_transpose"]),
            is_conjugate=bool(self.qualifiers[1]["is_conjugate"]),
        )
        c_layout = (
            MatrixLayout(shape=self.operands[2].shape, strides=self.operands[2].strides) if self.num_operands == 3 else None
        )

        input_layout = ("T" if a_layout.is_transpose else "N") + ("T" if b_layout.is_transpose else "N")
        if self.input_type_width <= 8 and input_layout != "TN":
            raise ValueError(f"FP8 matrix multiplications support only TN input layout. Got {input_layout}")

        # Get the operation traits.
        A_shape = problem_spec.shapes[0]  # this is global
        B_shape = problem_spec.shapes[1]  # this is global
        M0, K0 = (A_shape[0], A_shape[1]) if not a_layout.is_transpose else (A_shape[1], A_shape[0])
        K1, N0 = (B_shape[0], B_shape[1]) if not b_layout.is_transpose else (B_shape[1], B_shape[0])
        if K0 != K1:
            raise ValueError(
                f"The 'K' extent must match for the operands: K={K0} in operand A is not equal to K={K1} in operand B."
            )

        self.mm_traits = MMTraits(
            M=M0,
            N=N0,
            K=K0,
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=c_layout,
            d_layout=None,  # this is determined in plan
        )
        self.result_layout: None | MatrixLayout = None  # Wait till planning to determine this based on the epilog.
        self.logger.info(
            f"The matrix multiplication attributes are M={self.mm_traits.M}, N={self.mm_traits.N}, "
            f"K={self.mm_traits.K}, transA={a_layout.is_transpose} and transB={b_layout.is_transpose}."
        )

        def use_alt_cache():
            if distributions[0].process_grid == distributions[1].process_grid == distributions[2].process_grid:
                # cuBLASMp uses SUMMA if all the process grids are equal, but there are
                # cases that are problematic with SUMMA, so avoid SUMMA for now for those
                # cases by using alternate cache for one of the process grids.
                for d in distributions:
                    if isinstance(d, BlockNonCyclic) or d._is_1d_distribution():
                        return True
                nprow, npcol = distributions[0].process_grid.shape
                if (distributions[2].block_sizes[0] == M0 // nprow) or (distributions[2].block_sizes[1] == N0 // npcol):
                    return True
            return False

        # Create process grids.
        alt_cache = use_alt_cache()
        self.lib_process_grids = []
        with utils.device_ctx(self.device_id):
            for i, d in enumerate(distributions):
                grid = d.process_grid
                assert grid.layout is not None
                from_alt_cache = alt_cache and i == 2
                lib_grid = _grid_cache.get_library_process_grid(grid, self.device_id, nccl_comm, from_alt=from_alt_cache)
                self.lib_process_grids.append(lib_grid)
        self.logger.info("Created cuBLASMp process grids")

        # Set memory allocator.
        # cuBLASMp requires the workspace on NCCL symmetric heap.
        self.allocator = CublasMpMemoryManager(self.device_id, list(set(self.lib_process_grids)), self.logger)

        # Create and set the operation descriptor.
        self.mm_desc = cublasMp.matmul_descriptor_create(self.compute_type)
        self.mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)
        self.mm_desc_ifc.transa = (
            cublas.Operation.C
            if (a_layout.is_conjugate and a_layout.is_transpose)
            else cublas.Operation.T
            if a_layout.is_transpose
            else cublas.Operation.N
        )
        self.mm_desc_ifc.transb = (
            cublas.Operation.C
            if (b_layout.is_conjugate and b_layout.is_transpose)
            else cublas.Operation.T
            if b_layout.is_transpose
            else cublas.Operation.N
        )
        if self.options.sm_count_communication:
            self.mm_desc_ifc.communication_sm_count = self.options.sm_count_communication
        if self.options.algo_type:
            self.mm_desc_ifc.algo_type = self.options.algo_type

        if version == 800 and self.d_dtype_width <= 16 and self.compute_type.name.startswith("COMPUTE_32F"):
            # This is only needed for cuBLASMp 0.8 (versions >= 0.8.1 do this by default).
            self.mm_desc_ifc.communication_type = CudaDataType.CUDA_R_32F

        # This ensures cuBLASMp always returns full (reduced) bias gradient
        # output. This is independent of whether the output is partitioned
        # or replicated.
        self.mm_desc_ifc.bias_result_scheme = cublasMp.ResultScheme.FULL

        self.problem_spec = problem_spec

        # Planning preferences
        self.preferences = None

        # Epilog attributes.
        self.epilog = None

        # Epilog attributes: name-to-operand.
        self.epilog_operands: dict[str, typing.Any] = {}

        # Epilog attributes: epilog input name-to-handler.
        self.epilog_input_name_to_handler: dict[str, typing.Any] = {}

        # Epilog attributes: name-to-output tensor.
        self.epilog_outputs: dict[str, typing.Any] = {}

        # Keep track of epilog input traits for resetting operands.
        self.epilog_inputs_traits: dict[str, typing.Any] = {}

        # Keep track of epilog output handlers to allocate output in execute().
        self.epilog_output_handlers: list[EpilogOutputHandler] = []

        # Non-epilog aux outputs. Currently, only used for quantization outputs (amax etc.)
        self.aux_outputs: dict[str, typing.Any] | None = None

        # Plan attributes.
        self.matrix_descriptors: list[int] = []
        self.mm_planned = False

        # Workspace attributes.
        self.workspace_device: None | memory.MemoryPointer = None
        self.workspace_size_device = 0
        self.workspace_host: None | np.ndarray = None
        self.workspace_size_host = 0
        self.workspace_allocated_size = 0
        self.workspace_allocated_here = False

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

        # Track whether the user has called release_operands().
        self._operands_released = False
        # Device-side array with the quantization_scales
        self.quantization_scales_device: dict[str, utils.TensorHolder] = {}

        self.valid_state = True
        self.logger.info("The distributed Matmul operation has been created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _check_valid_matmul(self, *args, **kwargs):
        """
        Check if the Matmul object is alive and well.
        """
        if not self.valid_state:
            raise InvalidMatmulState("The Matmul object cannot be used after resources are free'd")

    def _check_valid_operands(self, *args, **kwargs):
        """
        Check if the operands are available for the operation.
        """
        what = kwargs["what"]
        if self._operands_released:
            raise RuntimeError(
                f"{what} cannot be performed after the operands have been released. "
                f"Use reset_operands() to provide new operands before performing the {what.lower()}."
            )

    def _free_plan_resources(self, exception: Exception | None = None) -> bool:
        """
        Free resources allocated in planning.
        """

        # Destroy matrix descriptors.
        for descriptor in self.matrix_descriptors:
            if descriptor is not None:
                cublasMp.matrix_descriptor_destroy(descriptor)
        self.matrix_descriptors = []

        self.mm_planned = False
        return True

    def _check_planned(self, *args, **kwargs):
        what = kwargs["what"]
        if not self.mm_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _free_workspace_memory(self, exception: Exception | None = None) -> bool:
        """
        Free workspace by releasing the MemoryPointer object.
        """
        if self.workspace_device is None:
            assert self.workspace_host is None, "Internal error."
            return True

        with utils.device_ctx(self.device_id):
            if self.workspace_stream is not None:
                # The workspace is on NCCL symmetric memory (allocated by cuBLASMp
                # with ncclMemAlloc).
                # Wait for the computation to finish before calling ncclMemFree.
                self.workspace_stream.sync()
            self.workspace_device.free()
        self.workspace_device = self.workspace_host = None
        self.workspace_allocated_size = 0
        self.logger.debug("[_free_workspace_memory] The workspace has been released.")

        return True

    def _reset_workspace_allocation_tracking(self):
        """
        Reset workspace allocation tracking attributes to False at the end of the methods
        where workspace memory is potentially allocated. This is necessary to prevent any
        exceptions raised before method entry from using stale tracking values.
        """
        self.workspace_allocated_here = False

    @utils.precondition(_check_valid_matmul)
    def _release_workspace_memory_perhaps(self, release_workspace):
        """
        Free workspace memory if it's larger than the specified limit.
        """
        if not release_workspace:
            return True

        # Establish ordering wrt the computation and free workspace if requested.
        if self.last_compute_event is not None:
            self.workspace_stream.wait(self.last_compute_event)
            self.logger.debug("Established ordering with respect to the computation before releasing the workspace.")
            self.last_compute_event = None

        self.logger.debug("[_release_workspace_memory_perhaps] The workspace memory will be released.")
        return self._free_workspace_memory()

    def _release_workspace_memory_perhaps_wrapper(self, exception: Exception | None = None) -> bool:
        """
        This is used in @atomic.
        """
        if isinstance(exception, cublasMp.cuBLASMpError) and (
            "NOT_SUPPORTED" in str(exception) or "INVALID_VALUE" in str(exception)
        ):
            addendum = (
                " It is also recommended to check the dtype support table at "
                "https://docs.nvidia.com/cuda/cublasmp/usage/functions.html#cublasmpmatmul"
            )
            # For cuBLASMpError we know that args attribute is (str,)
            exception.args = (exception.args[0] + addendum,)
        self._release_workspace_memory_perhaps(release_workspace=self.workspace_allocated_here)
        self._reset_workspace_allocation_tracking()
        return True

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator.
        """

        assert self.workspace_size_device is not None, "Internal Error."
        assert self.workspace_allocated_here is False, "Internal Error."

        if self.workspace_size_device == 0:  # For performance, bypass allocator for workspace size == 0.
            self.workspace_device = memory.MemoryPointer(0, 0, finalizer=None)
        else:
            self.logger.debug("Allocating device workspace for performing the matrix multiplication...")
            with utils.device_ctx(self.device_id), stream_holder.ctx:
                try:
                    if isinstance(self.allocator, memory.BaseCUDAMemoryManagerAsync):
                        self.workspace_device = self.allocator.memalloc_async(self.workspace_size_device, stream_holder.obj)
                    else:
                        self.workspace_device = self.allocator.memalloc(self.workspace_size_device)
                    self.workspace_allocated_here = True
                except TypeError as e:
                    message = (
                        "The method 'memalloc' in the allocator object must conform to the interface in the "
                        "'BaseCUDAMemoryManager' protocol."
                    )
                    raise TypeError(message) from e

        if self.workspace_size_host > 0:
            self.logger.debug("Allocating host workspace for performing the matrix multiplication...")
            self.workspace_host = np.array(self.workspace_size_host, dtype=np.int8)

        self.workspace_allocated_size = self.workspace_size_device
        self.workspace_stream = stream_holder.obj
        self.logger.debug(
            f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size_device)} in the context "
            f"of stream {self.workspace_stream}."
        )
        self.logger.debug(f"Finished allocating host workspace of size {formatters.MemoryStr(self.workspace_size_host)}.")

    def _allocate_workspace_memory_perhaps(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been
        done.
        """

        if self.workspace_device is not None and self.workspace_allocated_size >= self.workspace_size_device:
            return

        return self._allocate_workspace_memory(stream_holder)

    @utils.precondition(_check_valid_matmul)
    def _infer_blocking_sizes(self, problem_spec, m, k, n, epilog_AR):
        # Infer block sizes for the case of BlockNonCyclic 1D distributions with uniform
        # partition sizes. Even though the block sizes were set individually for each
        # distribution when BlockNonCyclic._bind() was called, the block sizes might need
        # to be tweaked because they have to match across matrices A, B and C/D for m, n, k
        # and so must be inferred jointly.
        if not all(isinstance(d, BlockNonCyclic) and d._is_1d_distribution() for d in self.distributions):
            return

        assert all(d._bound for d in self.distributions), "Internal error"

        nranks = self.nranks

        transA = self.mm_traits.a_layout.is_transpose
        transB = self.mm_traits.b_layout.is_transpose

        # This function only infers for uniform partition sizes.
        if self.distributions[0]._is_row_wise():
            if transA and k % nranks != 0:
                return
            if not transA and m % nranks != 0:
                return
        else:
            if transA and m % nranks != 0:
                return
            if not transA and k % nranks != 0:
                return

        if self.distributions[1]._is_row_wise():
            if transB and n % nranks != 0:
                return
            if not transB and k % nranks != 0:
                return
        else:
            if transB and k % nranks != 0:
                return
            if not transB and n % nranks != 0:
                return

        A = self.operands[0]
        B = self.operands[1]
        mbA, nbA = A.shape  # local
        mbB, nbB = B.shape  # local

        if epilog_AR:
            mbD, nbD = m, n
        else:
            mbD, nbD = (m // nranks, n) if self.distributions[2]._is_row_wise() else (m, n // nranks)

        # Note that for a dimension of length L that isn't partitioned, L//N is also
        # a valid block size (a single block in that dimension is equivalent to N
        # contiguous blocks in that dimension).
        if not transA:
            # A is (m, k)
            mbA = mbD = min(mbA, mbD)
            if not transB:
                # B is (k, n)
                nbA = mbB = min(nbA, mbB)
            else:
                # B is (n, k)
                nbA = nbB = min(nbA, nbB)
        else:
            # A is (k, m)
            nbA = mbD = min(nbA, mbD)
            if not transB:
                # B is (k, n)
                mbA = mbB = min(mbA, mbB)
            else:
                # B is (n, k)
                mbA = nbB = min(mbA, nbB)

        if not transB:
            # B is (k, n)
            nbB = nbD = min(nbB, nbD)
        else:
            # B is (n, k)
            mbB = nbD = min(mbB, nbD)

        self.distributions[0]._block_sizes = (mbA, nbA)
        self.distributions[1]._block_sizes = (mbB, nbB)
        self.distributions[2]._block_sizes = (mbD, nbD)

    @utils.precondition(_check_valid_matmul)
    def _infer_algo(self, m, k, n, epilog_AR: bool) -> int:
        """Return distributed matrix multiplication algorithm that is expected to run.
        Currently only tries to infer the tensor parallelism comm-overlap algorithms.

        0 if naive
        2 if GEMM+RS
        3 if AG+GEMM
        4 if GEMM+AR
        5 if local GEMM
        -1 otherwise (unknown, probably naive or SUMMA)
        """
        nranks = self.nranks
        if nranks == 1:
            return 5

        if any(not d._is_1d_distribution() for d in self.distributions):
            return -1  # unknown

        is_transpose = [
            self.mm_traits.a_layout.is_transpose,
            self.mm_traits.b_layout.is_transpose,
            False,
        ]

        global_shape = [
            (m, k) if not is_transpose[0] else (k, m),
            (k, n) if not is_transpose[1] else (n, k),
            (m, n) if not epilog_AR else (m * nranks, n) if self.distributions[2]._is_row_wise() else (m, n * nranks),
        ]

        # Check that data divides evenly and distribution is non-cyclic
        for i in range(3):
            partitioned_dim = 0 if self.distributions[i]._is_row_wise() else 1

            if global_shape[i][partitioned_dim] % nranks != 0:
                # Data has to divide evenly
                return -1

            block_sizes = self.distributions[i].block_sizes
            if global_shape[i][partitioned_dim] // nranks != block_sizes[partitioned_dim]:
                # has to be non-cyclic
                return -1

        transA = is_transpose[0]
        A_distribution = "R" if self.distributions[0]._is_row_wise() else "C"

        transB = is_transpose[1]
        B_distribution = "R" if self.distributions[1]._is_row_wise() else "C"

        C_distribution = "R" if self.distributions[2]._is_row_wise() else "C"

        expected_algo = 0  # naive
        if epilog_AR:
            expected_algo = 4  # GEMM+AR
        elif (
            C_distribution == "R" and A_distribution == ("C" if transA else "R") and B_distribution == ("R" if transB else "C")
        ):
            expected_algo = 3  # AG+GEMM
        elif (
            C_distribution == "C" and A_distribution == ("R" if transA else "C") and B_distribution == ("C" if transB else "R")
        ):
            expected_algo = 2  # GEMM+RS

        return expected_algo

    def _check_local_gemm_sizes_fp8(self):
        """Check that the local GEMM size is supported by cuBLASLt"""
        # The cuBLASMp algorithms considered here currently require uniform partitioning,
        # so the logic below assumes local GEMM size is the same on every rank.

        # Get m, n, k for local GEMM
        if self._expected_algo == 3:  # AG+GEMM
            # For local gemm:
            # - m is partitioned
            # - k and n are the global size (B is gathered)
            assert self.mm_traits.M % self.nranks == 0, "Internal error."
            m, n, k = self.mm_traits.M // self.nranks, self.mm_traits.N, self.mm_traits.K
        elif self._expected_algo in (2, 4):  # GEMM+RS, GEMM+AR
            # A and B are partitioned only on k
            assert self.mm_traits.K % self.nranks == 0, "Internal error."
            m, n, k = self.mm_traits.M, self.mm_traits.N, self.mm_traits.K // self.nranks
        elif self._expected_algo == 5:  # non-distributed GEMM
            m, n, k = self.mm_traits.M, self.mm_traits.N, self.mm_traits.K
        else:
            return

        if self.input_type_width == 8 and self.options.block_scaling and (m % 128 != 0 or n % 128 != 0 or k % 128 != 0):
            raise ValueError("Matrix sizes for local GEMM must be divisible by 128 when block_scaling=True.")

        if self.input_type_width == 8 and (m % 16 != 0 or n % 16 != 0 or k % 16 != 0):
            raise ValueError("Matrix sizes for local GEMM must be divisible by 16 for FP8 operations")

    def _validate_scalar_scale(self, operand: str):
        """
        Validates a scalar scale.
        """
        if self.options.block_scaling:
            raise ValueError(f"A scalar tensor-wide scale factor is not allowed for {operand.upper()} when block_scaling=True.")

    def _validate_tensor_scale(self, scale, operand: str, operand_size=None):
        """
        Validates a tensor scale.

        Args:
            scale: The tensor scale to validate
            operand: The operand name (a, b, c, d)
            operand_size: Size of the operand (needed for block scaling shape validation)
        """
        # Package validation: Normalize "numpy" to "cuda" to match the behavior in __init__.
        # When operands are NumPy on CPU, self.package is set to "cuda" (execution package),
        # so we must also normalize NumPy scales to "cuda" to allow the same input format.
        # This handles the NumPy <=> CuPy asymmetry where NumPy on CPU is accepted as input
        # but internally converted to CuPy for CUDA execution.
        scale_package = utils.infer_object_package(scale)
        scale_package = "cuda" if scale_package == "numpy" else scale_package
        if scale_package != self.package:
            raise TypeError(
                f"The quantization scaling tensor for {operand.upper()} must belong to the same package as the operands."
            )

        # Wrap temporarily since this is needed for validation
        scale_wrapped = tensor_wrapper.wrap_operand(scale)

        # Device/memory space validation
        expected_device_id = "cpu" if self.memory_space == "cpu" else self.device_id
        if expected_device_id != scale_wrapped.device_id:
            raise ValueError(
                f'The scale for {operand.upper()} is on device "{scale_wrapped.device_id}", '
                f'but it should be on device "{expected_device_id}" to match the operands memory space.'
            )

        # Shape and dtype validation for non-block-scaling
        if not self.options.block_scaling:
            if scale_wrapped.shape not in ((1,), ()):
                raise ValueError(
                    f"The scale for {operand.upper()} must be of shape (1,) or (). Got {scale_wrapped.shape} instead."
                )
            if scale_wrapped.dtype != "float32":
                raise ValueError(f"The scale for {operand.upper()} must be float32 type. Got {scale_wrapped.dtype} instead.")

        # Shape and dtype validation for block-scaling
        elif self.input_type_width == 8:
            # Dtype validation (always possible)
            if scale_wrapped.dtype != "uint8":
                raise ValueError(f"Block scales for {operand.upper()} must be uint8 tensor.")

            # Shape validation (only if operand_size is available)
            if operand_size is not None:
                expected_shape = (operand_size // 32,)
                if scale_wrapped.shape != expected_shape:
                    raise ValueError(
                        f"Scales for {operand.upper()} should have shape {expected_shape}. Got {scale_wrapped.shape}."
                    )
        else:
            raise ValueError("block_scaling == True is not supported for non-FP8 types.")

    def _validate_operand_scales(self, quantization_scales, all_required):
        """
        Validates quantization scales, wrapping them into a MatmulQuantizationScales
        object if needed.

        Args:
            quantization_scales: The quantization scales to validate.
            all_required: Whether all scales are required.

        Returns:
            A MatmulQuantizationScales object with the validated quantization scales.
        """
        if quantization_scales is None:
            raise ValueError(
                "Scales are required for narrow-precision (FP8 and lower) operations. "
                "Please set `quantization_scales` argument."
            )

        # wrap the quantization scales into a MatmulQuantizationScales object if needed
        # otherwise, return the quantization scales as is
        quantization_scales = utils.check_or_create_options(
            _configuration.MatmulQuantizationScales, quantization_scales, "Scale factors"
        )

        # Validate which scales are required/allowed
        expected_scales = "AB"
        if self.d_dtype_width <= 8 and not self.options.block_scaling:
            expected_scales += "D"
        elif quantization_scales.d is not None:
            if self.options.block_scaling:
                raise ValueError("Quantization scaling is not supported for D when `block_scaling` option is enabled.")
            if self.d_dtype_width > 8:
                raise ValueError(
                    "Quantization scaling is not supported for D when it is not a narrow-precision (FP8 and lower) type."
                )

        if self.num_operands == 3 and self.c_dtype_width <= 8:
            expected_scales += "C"
        elif quantization_scales.c is not None:
            raise ValueError(
                "Quantization scaling is not supported for C when it is not a narrow-precision (FP8 and lower) type."
            )

        if all_required:
            for operand in expected_scales:
                if getattr(quantization_scales, operand.lower()) is None:
                    raise ValueError(f"Scale for {operand.upper()} is not specified")

        # Validate each scale by delegating to scalar/tensor specific validators
        for operand in "abcd":
            scale = getattr(quantization_scales, operand)
            if scale is None:
                continue

            if isinstance(scale, int | float):
                self._validate_scalar_scale(operand)
            else:
                # For block scaling, pass operand size for shape validation
                if self.options.block_scaling and operand in ("a", "b"):
                    operand_idx = 0 if operand == "a" else 1
                    operand_size = self.operands[operand_idx].size  # type: ignore[union-attr,index]
                else:
                    operand_size = None
                self._validate_tensor_scale(scale, operand, operand_size)

        return quantization_scales

    def _validate_epilog_aux_scale(self, aux_quantization_scale, *, required):
        is_fp8_aux = (
            self.preferences.epilog.aux_type is not None
            and typemaps.NAME_TO_DATA_WIDTH[typemaps.DATA_TYPE_TO_NAME[self.preferences.epilog.aux_type]] <= 8
        )
        if aux_quantization_scale is not None and not is_fp8_aux:
            raise ValueError(
                "Scales for epilog auxiliary output are not supported when `preferences.epilog.aux_type` is not set to a "
                "narrow-precision type."
            )
        elif aux_quantization_scale is None and is_fp8_aux and required:
            raise ValueError(
                '"aux_quantization_scale" epilog input is required when `preferences.epilog.aux_type` is set to a '
                "narrow-precision type."
            )

        # Validate scalar vs tensor scale (same as for operand scales)
        if aux_quantization_scale is not None:
            if isinstance(aux_quantization_scale, int | float):
                self._validate_scalar_scale("epilog_aux")
            else:
                # No operand_size for epilog_aux scales
                self._validate_tensor_scale(aux_quantization_scale, "epilog_aux", operand_size=None)

    def _prepare_validated_scalar_scale(self, scale: int | float, operand: str, stream_holder: utils.StreamHolder):
        """
        Converts validated scalar to float32 tensor and copies to GPU.
        Assumes validation already done in _validate_scalar_scale.
        """
        # If it's a scalar, copy to GPU. Float32 is the only type allowed by
        # cublasLtMatmulScale_t for tensor-wide scaling.
        self.logger.debug(f"Scale for {operand.upper()} will be copied to device {self.device_id}.")
        scale_op = tensor_wrapper.wrap_operand(np.asarray([scale], dtype="float32"))
        self.quantization_scales_device[operand] = scale_op.to(self.device_id, stream_holder)

    def _prepare_validated_tensor_scale(self, scale, operand: str, stream_holder: utils.StreamHolder):
        """
        Wraps validated tensor and copies to GPU.
        Assumes all validation already done in _validate_tensor_scale (called in __init__).
        This is pure preparation - no validation here.

        Note: We wrap the tensor a second time here (first wrap was for validation).
        This is acceptable because wrapping is cheap and we get early error detection.
        """
        # Wrap the scale (second time - first was for validation in __init__)
        self.quantization_scales_device[operand] = tensor_wrapper.wrap_operand(scale)

        # Copy to GPU if on CPU (no validation, just preparation)
        if self.quantization_scales_device[operand].device in (None, "cpu"):
            self.logger.debug(f"Scale for {operand.upper()} will be copied to device {self.device_id}.")
            self.quantization_scales_device[operand] = self.quantization_scales_device[operand].to(
                self.device_id, stream_holder
            )

    def _prepare_single_validated_scale(self, scale, operand: str, cublas_operand: str, stream_holder: utils.StreamHolder):
        """
        Prepares a single validated scale and sets its pointer/mode in mm_desc_ifc.
        Used for both operand scales (a,b,c,d) and epilog scales (epilog_aux).
        Assumes validation already done.
        """
        if scale is None:
            return

        # Delegate to specific preparer (validation already done)
        if isinstance(scale, int | float):
            self._prepare_validated_scalar_scale(scale, operand, stream_holder)
        else:
            self._prepare_validated_tensor_scale(scale, operand, stream_holder)

        # Set pointer and mode in descriptor
        setattr(self.mm_desc_ifc, f"{cublas_operand}_scale_pointer", self.quantization_scales_device[operand].data_ptr)

        if self.options.block_scaling:
            self.logger.debug(f"Using VEC32_UE8M0 scale mode for operand {operand.upper()}.")
            setattr(self.mm_desc_ifc, f"{cublas_operand}_scale_mode", cublasMp.MatmulMatrixScale.VEC32_UE8M0)
        else:
            self.logger.debug(f"Using SCALAR_32F scale mode for operand {operand.upper()}.")
            setattr(self.mm_desc_ifc, f"{cublas_operand}_scale_mode", cublasMp.MatmulMatrixScale.SCALAR_FP32)

    def _prepare_operand_quantization_scales(self, scales, stream_holder: utils.StreamHolder):
        """
        Prepares validated operand scales (a,b,c,d).
        Assumes scales are validated and wrapped into a MatmulQuantizationScales object.
        """
        for operand in "abcd":
            scale = getattr(scales, operand)
            self._prepare_single_validated_scale(scale, operand, cublas_operand=operand, stream_holder=stream_holder)

    @utils.precondition(_check_valid_matmul)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(
        self, *, preferences=None, epilog=None, epilog_inputs=None, stream: utils.AnyStream | int | None = None
    ):  # Epilog inputs require as many inputs (with specific shapes etc) as required by the epilogue. It's a dict.
        """
        Plan the matrix multiplication operation, considering the epilog (if provided).

        Args:
            preferences: {preferences}

            epilog: {epilog}

            epilog_inputs: {epilog_inputs}

            stream: {stream}

        Notes:
            Epilogs that have ``BIAS`` in their name need an epilog input with the key
            ``'bias'``. Epilogs that have ``DRELU`` need an epilog input with the key
            ``'relu_aux'``, which is produced in a "forward pass" epilog like ``RELU_AUX``
            or ``RELU_AUX_BIAS``. Similarly, epilogs with ``DGELU`` in their name require an
            epilog input with the key ``'gelu_aux'``, produced in the corresponding forward
            pass operation.

        See :class:`Matmul` for an example, and further examples can be found in the
        `nvmath/examples/distributed/linalg/advanced/matmul
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_
        directory.
        """
        self.logger.info("= PLANNING PHASE =")

        # Clear epilog operands, since different epilogs can be provided in different calls.
        # We don't need to worry about ordering, since it's the user's responsibility to
        # order calls that accept a stream argument. This applies to CPU operands as well,
        # even though we move them to the GPU, since the execution is blocking.
        self.epilog_operands = {}  # Clear operands in case of repeated planning.
        self.epilog_input_name_to_handler = {}  # Clear input name to handler map as well,
        self.epilog_inputs_traits = {}  # ... and the input traits as well.

        preferences = utils.check_or_create_options(
            _configuration.MatmulPlanPreferences, preferences, "Distributed matrix multiplication plan preferences"
        )
        self.preferences = preferences

        if self._operands_released:
            raise RuntimeError("The Matmul has no operands. Please call reset_operands")

        mm_traits = self.mm_traits

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for the matrix multiplication plan is {stream_holder.obj}.")

        if epilog is None and epilog_inputs is not None:
            self.logger.warning(
                f"Matmul: The provided epilog inputs {epilog_inputs.keys()} are ignored since an epilog is not specified."
            )

        m, n, k = mm_traits.M, mm_traits.N, mm_traits.K

        if self.num_operands == 3:
            if epilog == MatmulEpilog.ALLREDUCE:
                expected_global_shape = tuple(
                    x * y for x, y in zip(self.distributions[2].process_grid.shape, (m, n), strict=False)
                )
                if tuple(self.problem_spec.shapes[2]) != expected_global_shape:
                    raise ValueError(
                        f"The global shape of C according to its distribution ({self.problem_spec.shapes[2]}) is "
                        f"not the expected one when using AllReduce epilogue ({expected_global_shape})"
                    )
                if self.operands[2].shape != (m, n):
                    raise ValueError(f"The shape of C on every process when using AllReduce epilogue must be (m, n)={(m, n)}")
            elif tuple(self.problem_spec.shapes[2]) != (m, n):
                raise ValueError(
                    f"The global shape of C according to its distribution ({self.problem_spec.shapes[2]}) is "
                    f"not the expected shape ({(m, n)})"
                )

        if not self.distributions[0]._bound:
            for i, d in enumerate(self.distributions):
                assert not d._bound, "Internal error"
                if i < self.num_operands:
                    global_shape = tuple(self.problem_spec.shapes[i])
                    shape = self.operands[i].shape
                else:
                    if epilog == MatmulEpilog.ALLREDUCE:
                        global_shape = tuple(
                            x * y for x, y in zip(self.distributions[2].process_grid.shape, (m, n), strict=False)
                        )
                    else:
                        global_shape = (m, n)
                    shape = None
                d._bind(global_shape, shape=shape)
            self._infer_blocking_sizes(self.problem_spec, m, k, n, epilog == MatmulEpilog.ALLREDUCE)

        transA = self.mm_traits.a_layout.is_transpose
        transB = self.mm_traits.b_layout.is_transpose

        # Check block size on m dimension.
        m_block_size_A = self.distributions[0].block_sizes[1] if transA else self.distributions[0].block_sizes[0]
        m_block_size_D = self.distributions[2].block_sizes[0]
        if m_block_size_A != m_block_size_D:
            raise ValueError("Block size of m dimension must be the same for A and C/D")

        # Check block size on n dimension.
        n_block_size_B = self.distributions[1].block_sizes[0] if transB else self.distributions[1].block_sizes[1]
        n_block_size_D = self.distributions[2].block_sizes[1]
        if n_block_size_B != n_block_size_D:
            raise ValueError("Block size of n dimension must be the same for B and C/D")

        # Check block size on k dimension.
        k_block_size_A = self.distributions[0].block_sizes[0] if transA else self.distributions[0].block_sizes[1]
        k_block_size_B = self.distributions[1].block_sizes[1] if transB else self.distributions[1].block_sizes[0]
        if k_block_size_A != k_block_size_B:
            raise ValueError("Block size of k dimension must be the same for A and B")

        self._expected_algo = self._infer_algo(m, k, n, epilog == MatmulEpilog.ALLREDUCE)

        # Fill the result traits, now that we know the epilog.
        result_shape = self.distributions[2]._data_shape
        self.result_layout = MatrixLayout(
            shape=result_shape,
            strides=calculate_strides(result_shape, (0, 1)) if not self.inplace else self.operands[2].strides,
        )
        mm_traits.d_layout = self.result_layout

        self.epilog = epilog
        if epilog is not None:
            assert epilog in EPILOG_INPUT_HANDLERS_MAP, "Not supported."
            self.logger.info(f"The specified epilog is {epilog.name}.")

            epilog_minimum_versions = EPILOG_MINIMUM_VERSIONS_MAP[epilog]
            version = cublasMp.get_version()
            if version < epilog_minimum_versions["cublasMp"]:
                message = (
                    f"The epilog {epilog.name} requires cublasMp >= {epilog_minimum_versions['cublasMp']}; "
                    f"you have version {version}."
                )
                raise ValueError(message)

            # Take a copy of the user-provided inputs.
            if epilog_inputs is not None:
                epilog_inputs = epilog_inputs.copy()
            else:
                epilog_inputs = {}

            # Get the dtype of auxiliary buffer
            aux_dtype_name = (
                typemaps.DATA_TYPE_TO_NAME[self.preferences.epilog.aux_type]  # type: ignore[attr-defined]
                if self.preferences.epilog.aux_type is not None  # type: ignore[attr-defined]
                else None
            )

            # Extract aux quantization scale from the inputs.
            aux_quantization_scale = (
                epilog_inputs.pop("aux_quantization_scale") if "aux_quantization_scale" in epilog_inputs else None
            )
            self._validate_epilog_aux_scale(aux_quantization_scale, required=True)
            self._prepare_single_validated_scale(
                aux_quantization_scale, "epilog_aux", cublas_operand="epilogue_aux", stream_holder=stream_holder
            )

            epilog_input_handler_types = EPILOG_INPUT_HANDLERS_MAP[epilog]
            if epilog_input_handler_types:
                epilog_input_handlers = [
                    handler_type(self.logger, mm_traits, epilog, self.c_dtype_name, self.d_dtype_name, aux_dtype_name)
                    for handler_type in epilog_input_handler_types
                ]

                required_epilog_input_names = {h.name for h in epilog_input_handlers}

                self.logger.info(f"The epilog requires the following additional inputs: {required_epilog_input_names}.")
                if required_epilog_input_names != set(epilog_inputs.keys()):
                    raise ValueError(
                        f"The epilog {epilog.name} requires the following input tensors: "
                        f"{required_epilog_input_names}. The provided tensor names are: {epilog_inputs.keys()}"
                    )

                # Wrap epilog inputs.
                for name in epilog_inputs:
                    epilog_inputs[name] = nvmath.internal.tensor_wrapper.wrap_operand(epilog_inputs[name])

                # Check if epilog inputs all belong to the same package, which is the same
                # as the package of the MM operands.
                epilog_package = utils.get_operands_package(list(epilog_inputs.values()))
                epilog_package = "cuda" if epilog_package == "numpy" else epilog_package  # Handle the NumPy <=> CuPy asymmetry.
                if self.package != epilog_package:
                    message = f"Library package mismatch for epilog: '{self.package}' => '{epilog_package}'"
                    raise TypeError(message)

                # Check if all epilog inputs all are on the same device, which is the device
                # of the operands.
                device_id = utils.get_operands_device_id(list(epilog_inputs.values()))
                if device_id != "cpu" and self.device_id != device_id:
                    raise ValueError(
                        f"The epilog inputs must be on the same device ({device_id}) as the operands ({self.device_id})."
                    )

                # Move epilog inputs to the GPU, if needed.
                if device_id == "cpu":
                    for e in required_epilog_input_names:
                        self.logger.debug(f"The epilog input {e} will be copied to device{self.device_id}.")
                        self.epilog_operands[e] = epilog_inputs[e].to(self.device_id, stream_holder)
                else:
                    for e in required_epilog_input_names:
                        self.epilog_operands[e] = epilog_inputs[e]

                # First validate all epilog inputs. Use the GPU tensors in case metadata has
                # changed.
                for handler in epilog_input_handlers:
                    handler.validate(epilog_inputs[handler.name])

                # Finally, update the MM descriptor. Note that we pass in
                # self.epilog_operands (which are on the GPU).
                for handler in epilog_input_handlers:
                    handler.update(self.mm_desc_ifc, self.epilog_operands[handler.name])
                    self.epilog_input_name_to_handler[handler.name] = handler

                # Capture the epilog operands traits for consistency checks when resetting
                # operands.
                self.epilog_inputs_traits = {
                    name: EpilogInputTraits(
                        dtype=self.epilog_operands[name].dtype,
                        extents=self.epilog_operands[name].shape,
                        strides=self.epilog_operands[name].strides,
                    )
                    for name in self.epilog_operands
                }

            epilog_output_handler_types = EPILOG_OUTPUT_HANDLERS_MAP[epilog]
            if epilog_output_handler_types:
                self.epilog_output_handlers = epilog_output_handlers = [
                    handler_type(self.logger, mm_traits, epilog, self.c_dtype_name, self.d_dtype_name, aux_dtype_name)
                    for handler_type in epilog_output_handler_types
                ]

                # Update the MM descriptor, except for the device pointer.
                for ohandler in epilog_output_handlers:
                    ohandler.update(self.mm_desc_ifc)

            # Set the epilog. At this point, we're sure that the epilog inputs, if any, are
            # valid and have been set.
            self.mm_desc_ifc.epilogue = epilog

        # Create descriptors for matrices A, B, C and D.
        matrix_dtypes = (self.a_dtype, self.b_dtype, self.c_dtype, self.d_dtype)
        for i in range(4):
            distribution = self.distributions[min(i, 2)]  # distribution for C/D is the same
            lld = self.operands[i].strides[1] if i < self.num_operands else distribution._data_shape[0]
            if i == 3 and self.inplace:
                lld = self.operands[2].strides[1]
            descriptor = cublasMp.matrix_descriptor_create(
                distribution._data_global_shape[0],
                distribution._data_global_shape[1],
                distribution.block_sizes[0],
                distribution.block_sizes[1],
                distribution.first_process[0],
                distribution.first_process[1],
                lld,
                matrix_dtypes[i],
                self.lib_process_grids[min(i, 2)],
            )
            self.matrix_descriptors.append(descriptor)

        if self.options.block_scaling and self.d_dtype_width == 8:
            self.mm_desc_ifc.d_out_scale_mode = cublasMp.MatmulMatrixScale.VEC32_UE8M0

        if self.input_type_width <= 8:
            self._check_local_gemm_sizes_fp8()
            self._prepare_operand_quantization_scales(self.quantization_scales, stream_holder)

        alpha_ptr, beta_ptr = self.alpha.ctypes.data, self.beta.ctypes.data
        self.workspace_size_device, self.workspace_size_host = cublasMp.matmul_buffer_size(
            self.handle,
            self.mm_desc,
            mm_traits.M,
            mm_traits.N,
            mm_traits.K,
            alpha_ptr,
            self.operands[0].data_ptr,
            1,
            1,
            self.matrix_descriptors[0],
            self.operands[1].data_ptr,
            1,
            1,
            self.matrix_descriptors[1],
            beta_ptr,
            0 if self.num_operands == 2 else self.operands[2].data_ptr,
            1,
            1,
            self.matrix_descriptors[2],
            0,  # d pointer
            1,
            1,
            self.matrix_descriptors[3],
        )

        self.mm_planned = True

    def _set_result_sheap_flag(self):
        self.result_on_symmetric_memory = False
        on_symmetric_memory = {o.is_symmetric_memory for o in self.operands}
        if len(on_symmetric_memory) == 2:
            self.logger.warning(
                "Some operands are on symmetric memory and others are not. Result won't be allocated on symmetric memory"
            )
        elif on_symmetric_memory == {True}:
            if self.memory_space == "cuda":
                self.logger.info("Input operands are on symmetric memory. Result will be allocated on symmetric memory.")
            self.result_on_symmetric_memory = True

    def _check_and_set_operand(
        self,
        operand,
        operand_name,
        mm_desc_ifc,
        stream_holder,
        *,
        operand_index=None,
        epilog_name=None,
        package=None,
        dtype=None,
        extents=None,
        strides=None,
    ):
        """
        Check to make sure that the provided operand is consistent with the one it's
        updating, and update it.
        """
        assert (operand_index is None) ^ (epilog_name is None), "Internal Error."
        assert self.operands is not None, "Internal Error."

        # Make sure that the data type and extents match.
        utils.check_attribute_match(dtype, operand.dtype, "data type")
        utils.check_attribute_match(extents, operand.shape, "extents")

        package = utils.infer_object_package(operand.tensor)

        # Conjugate flag of the provided operands must match the original qualifiers
        if operand_index is not None and self.lazy_conjugation[operand_index] != operand.is_conjugate:
            raise ValueError(f"The provided operand {operand_name} has different conjugate flag than the original operand")

        device_id = operand.device_id
        # When memory_space is "cpu", operands should be on CPU even though
        # self.device_id is the execution device.
        expected_device_id = "cpu" if self.memory_space == "cpu" else self.device_id
        if expected_device_id != device_id:
            raise ValueError(
                f'The operand {operand_name} is on device "{device_id}", but it should be on device '
                f'"{expected_device_id}" to match the original operand.'
            )

        if device_id == "cpu":
            package = "cuda" if package == "numpy" else package  # Handle the NumPy <=> CuPy asymmetry.
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            # Check if we have a GPU buffer to update into.
            if operand_index is not None:
                o = self.operands[operand_index]
            else:
                o = self.epilog_operands[epilog_name]
            if o is None:  # No buffer, create one.
                # Copy operand across memory spaces (CPU to GPU).
                # Some of the comm overlap algorithms in cuBLASMp will perform better when
                # some of the operands are already on symmetric memory (e.g. AG+GEMM when
                # B is on symmetric memory).
                o = operand.to(self.device_id, stream_holder)
                if operand_index is not None:
                    self.operands[operand_index] = o
                else:
                    self.epilog_operands[epilog_name] = o
                    # Update the epilog pointer, since we're starting afresh.
                    self.epilog_input_name_to_handler[epilog_name].update(mm_desc_ifc, o)
            else:
                # In-place copy to existing device pointer because the new operand is on the
                # CPU.
                o.copy_(operand, stream_holder=stream_holder)
            if self.memory_space == "cpu" and operand_index == 2:
                # Hold references, needed for inplace operations.
                self.cpu_c_ref = operand
        else:
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            utils.check_attribute_match(strides, operand.strides, "strides")

            if self.device_id != device_id:
                raise ValueError(
                    f"The operand {operand_name} must be on the same device ({device_id}) as the original operand "
                    f"({self.device_id})."
                )

            # Finally, replace the original operand by the new one.
            if operand_index is not None:
                self.operands[operand_index] = operand
            else:
                self.epilog_operands[epilog_name] = operand
                # Update the epilog pointer, since we're starting afresh.
                self.epilog_input_name_to_handler[epilog_name].update(mm_desc_ifc, operand)

        self.logger.info(f"Operand '{operand_name}' has been reset to the new value.")

        return

    @utils.precondition(_check_valid_matmul)
    def reset_operands(
        self,
        *,
        a=None,
        b=None,
        c=None,
        alpha=None,
        beta=None,
        quantization_scales=None,
        epilog_inputs=None,
        stream: utils.AnyStream | int | None = None,
    ):
        """
        Reset one or more operands held by this :class:`Matmul` instance.
        Only the operands explicitly passed are updated; omitted operands retain
        their current values.

        This method will perform various checks on the new operands to make sure:

        - The distributions, shapes, strides, datatypes match those of the old ones.
        - The packages that the operands belong to match those of the old ones.
        - If input tensors are on GPU, the device must match.

        .. versionchanged:: 0.9
            All parameters are now keyword-only.

        Args:
            a: {a}

            b: {b}

            c: {c}

            alpha: {alpha}

            beta: {beta}

            quantization_scales: {quantization_scales}

            epilog_inputs: {epilog_inputs}

            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import nvmath.distributed
            >>> from nvmath.distributed.distribution import Slab

            Get process group used to initialize nvmath.distributed (for information on
            initializing ``nvmath.distributed``, you can refer to the documentation or to
            the Matmul examples in `nvmath/examples/distributed/linalg/advanced
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_):

            >>> process_group = nvmath.distributed.get_context().process_group

            Get my process rank:

            >>> rank = process_group.rank

            Create two 3-D float64 ndarrays on the GPU (using Slab distributions to
            distribute the matrices across processes):

            >>> M, N, K = 128, 128, 256
            >>> a_shape = Slab.X.shape(rank, (M, K))
            >>> b_shape = Slab.Y.shape(rank, (K, N))
            >>> device_id = nvmath.distributed.get_context().device_id
            >>> with cp.cuda.Device(device_id):
            ...     a = cp.asfortranarray(cp.random.rand(*a_shape))
            ...     b = cp.asfortranarray(cp.random.rand(*b_shape))

            Create an matrix multiplication object as a context manager

            >>> d = [Slab.X, Slab.Y, Slab.X]
            >>> with nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=d) as mm:
            ...     # Plan the operation.
            ...     mm.plan()
            ...
            ...     # Execute the MM to get the first result.
            ...     r1 = mm.execute()
            ...
            ...     # Reset the operands to new CuPy ndarrays.
            ...     with cp.cuda.Device(device_id):
            ...         a_new = cp.asfortranarray(cp.random.rand(*a_shape))
            ...         b_new = cp.asfortranarray(cp.random.rand(*b_shape))
            ...     mm.reset_operands(a=a_new, b=b_new)
            ...
            ...     # Execute to get the new result corresponding to the updated operands.
            ...     r2 = mm.execute()

            Note that if only a subset of operands are reset, the operands that are not
            reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is
            functionally equivalent to updating the operands in-place, i.e, replacing
            ``mm.reset_operands(a=a_new, b=b_new)`` with ``a[:]=a_new`` and ``b[:]=b_new``.
            Note that updating the operand in-place should be adopted with caution as
            it can only yield the expected result when the operand memory space is
            accessible from the execution space.

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul/example06_stateful_inplace.py>`_.

        .. seealso::
            :meth:`release_operands`
        """

        if c is not None and self.num_operands == 2:
            raise ValueError(
                "The matrix multiplication problem specification does not include operand C, so it cannot be reset."
            )

        # If operands have been released, all required operands must be provided.
        if self._operands_released:
            abc_ok = a is not None and b is not None and (self.num_operands != 3 or c is not None)

            epilog_names = self.epilog_inputs_traits.keys()
            epilog_ok = True
            if epilog_names:
                if epilog_inputs is None:
                    epilog_ok = False
                elif epilog_names != epilog_inputs.keys():
                    raise ValueError(
                        f"The epilog inputs {epilog_names} are required. The provided epilog input names are "
                        f"{epilog_inputs.keys()}."
                    )

            scales_ok = True
            needs_scales = self.input_type_width <= 8
            if needs_scales and quantization_scales is None:
                scales_ok = False

            if not abc_ok:
                raise ValueError(
                    "After release_operands(), all required operands must be provided to reset_operands(). "
                    f"Required: a, b{', c' if self.num_operands == 3 else ''}."
                )
            if not scales_ok:
                raise ValueError("After release_operands(), quantization_scales must be provided to reset_operands().")
            if not epilog_ok:
                raise ValueError(
                    "After release_operands(), epilog_inputs must be provided to reset_operands(). "
                    f"Required epilog inputs: {epilog_names}."
                )

            self.operands = [None] * self.num_operands  # type: ignore
            self.epilog_operands = dict.fromkeys(epilog_names)
            if needs_scales:
                self.quantization_scales = _configuration.MatmulQuantizationScales()

        # Update alpha.
        if alpha is not None:
            try:
                self.alpha[0] = alpha
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'."
                ) from e

        # Update beta.
        if beta is not None:
            if self.num_operands == 2:
                self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
            else:
                try:
                    self.beta[0] = beta
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'."
                    ) from e

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)

        # Update quantization_scales.
        if quantization_scales is not None:
            quantization_scales = self._validate_operand_scales(quantization_scales, all_required=False)
            if quantization_scales.a is not None:
                self.quantization_scales.a = quantization_scales.a
            if quantization_scales.b is not None:
                self.quantization_scales.b = quantization_scales.b
            if quantization_scales.c is not None:
                self.quantization_scales.c = quantization_scales.c
            if quantization_scales.d is not None:
                self.quantization_scales.d = quantization_scales.d
            self._prepare_operand_quantization_scales(self.quantization_scales, stream_holder)

        if epilog_inputs is not None and "aux_quantization_scale" in epilog_inputs:
            epilog_inputs = epilog_inputs.copy()
            aux_quantization_scale = epilog_inputs.pop("aux_quantization_scale")
            self._validate_epilog_aux_scale(aux_quantization_scale, required=False)
            self._prepare_single_validated_scale(
                aux_quantization_scale, "epilog_aux", cublas_operand="epilogue_aux", stream_holder=stream_holder
            )

        # Reset the provided operands.
        if a is not None:
            a = tensor_wrapper.wrap_operand(a)
            index = 0
            self._check_and_set_operand(
                a,
                "A",
                self.mm_desc_ifc,
                stream_holder,
                operand_index=index,
                dtype=self.a_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

        if b is not None:
            b = tensor_wrapper.wrap_operand(b)
            index = 1
            self._check_and_set_operand(
                b,
                "B",
                self.mm_desc_ifc,
                stream_holder,
                operand_index=index,
                dtype=self.b_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

        if c is not None:  # If we get here, we know that C is one of the operands in the problem specification.
            c = tensor_wrapper.wrap_operand(c)
            index = 2
            self._check_and_set_operand(
                c,
                "C",
                self.mm_desc_ifc,
                stream_holder,
                operand_index=index,
                dtype=self.c_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

        # Reset the provided epilog inputs.
        if epilog_inputs is not None:
            for name in epilog_inputs:
                epilog_input = tensor_wrapper.wrap_operand(epilog_inputs[name])
                self._check_and_set_operand(
                    epilog_input,
                    name,
                    self.mm_desc_ifc,
                    stream_holder,
                    epilog_name=name,
                    dtype=self.epilog_inputs_traits[name].dtype,
                    extents=self.epilog_inputs_traits[name].extents,
                    strides=self.epilog_inputs_traits[name].strides,
                )

        self._set_result_sheap_flag()
        self._operands_released = False

    @utils.precondition(_check_valid_matmul)
    def release_operands(self):
        """
        {release_operands}
        """
        if self._operands_released:
            self.logger.info("Operands have already been released; nothing to do.")
            return

        # CUDA memory space:
        #   self.operands, self.epilog_operands, and
        #   self.quantization_scales_device hold direct user references;
        #   self.quantization_scales holds the user-provided scales object.
        # CPU memory space:
        #   self.operands, self.epilog_operands, and
        #   self.quantization_scales_device hold internal device mirrors;
        #   self.quantization_scales holds the user-provided scales object;
        #   self.cpu_c_ref holds a direct reference to the user's CPU operand C.
        # In both cases, release all of them.
        # Note that if/when possible, we keep the TensorHolder wrappers and
        # container structures alive and only release the internal tensor reference.
        # This is useful when reset_operands_unchecked is called subsequently
        # because it can reuse the existing wrappers, saving overhead.
        for op in self.operands:
            op.tensor = None
        for op in self.epilog_operands.values():
            op.tensor = None
        for op in self.quantization_scales_device.values():
            op.tensor = None

        # The attribute itself might not exist (non-FP8 paths), so check first.
        if hasattr(self, "quantization_scales"):
            self.quantization_scales.a = None
            self.quantization_scales.b = None
            self.quantization_scales.c = None
            self.quantization_scales.d = None

        if self.memory_space == "cpu" and self.cpu_c_ref is not None:
            self.cpu_c_ref.tensor = None

        self._operands_released = True
        self.logger.info("User-provided operands have been released.")

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operands, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps_wrapper, method=True)
    def execute(self, *, release_workspace=False, stream: utils.AnyStream | int | None = None):
        """
        Execute a planned distributed matrix multiplication.

        Args:
            release_workspace: {release_workspace}

            stream: {stream}

        Returns:
           {result}
        """
        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        assert self.operands is not None, "Internal error."

        if log_info:
            self.logger.info("= EXECUTION PHASE =")
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        if log_info:
            self.logger.info(f"The specified stream for execute() is {stream_holder.obj}.")

        # Allocate workspace if needed.
        # Note: workspace is allocated with cuBLASMp allocator, which calls ncclMemAlloc
        # (which is blocking).
        self._allocate_workspace_memory_perhaps(stream_holder)

        # Create empty tensors for auxiliary output.
        for handler in self.epilog_output_handlers:
            name = handler.name
            shape, strides, dtype_name = handler.attributes()
            if log_debug:
                self.logger.debug(f"Beginning auxiliary output tensor '{name}' creation...")
                self.logger.debug(f"The '{name}' tensor shape = {shape} with strides = {strides} and data type '{dtype_name}'.")
            self.epilog_outputs[name] = aux_tensor = utils.create_empty_tensor(
                self.result_class,
                shape,
                dtype_name,
                self.device_id,
                stream_holder,
                verify_strides=False,
                strides=strides,
            )
            if name == "relu_aux" and self._expected_algo == 3:
                with utils.device_ctx(self.device_id):
                    aux_tensor.tensor[:] = 0
            if log_debug:
                self.logger.debug(f"The auxiliary output tensor '{name}' has been created.")
            if self.preferences.epilog.aux_amax:  # type: ignore[attr-defined]
                if "float8" not in dtype_name:
                    raise ValueError("epilog.aux_amax=True is not supported when epilog output type is not FP8.")
                self.epilog_outputs[f"{name}_amax"] = utils.create_empty_tensor(
                    self.result_class,
                    (1,),
                    "float32",  # This is the only type allowed by cuBLAS for AMAX.
                    self.device_id,
                    stream_holder,
                    verify_strides=False,
                )
                self.mm_desc_ifc.epilogue_aux_amax_pointer = self.epilog_outputs[f"{name}_amax"].data_ptr

            # Update the data pointer in the MM descriptor.
            handler.update_ptr(self.mm_desc_ifc, aux_tensor.data_ptr)

        # Create empty tensor for the result, if the operation is not in-place.
        # result_layout is based on local properties.
        assert self.result_layout is not None, "Internal Error. self.result_layout should have been set by self.plan()"
        if self.inplace:
            if log_debug:
                self.logger.debug("The operation is in-place (operand C will be overwritten).")
            result = self.operands[2]
        else:
            if log_debug:
                self.logger.debug("Beginning output (empty) tensor creation...")
                self.logger.debug(
                    f"The local output tensor shape = {self.result_layout.shape} with strides = "
                    f"{self.result_layout.strides} and data type '{self.d_dtype_name}'."
                )
            result = cast(
                DistributedTensor,
                utils.create_empty_tensor(
                    self.result_class,
                    self.result_layout.shape,
                    self.d_dtype_name,
                    self.device_id,
                    stream_holder,
                    verify_strides=False,
                    strides=self.result_layout.strides,
                    symmetric_memory=self.result_on_symmetric_memory,
                    make_symmetric=self.result_on_symmetric_memory,
                ),
            )
            if log_debug:
                self.logger.debug("The output (empty) tensor has been created.")

        self.aux_outputs = {}

        if self.options.result_amax:
            self.aux_outputs["result_amax"] = utils.create_empty_tensor(
                self.result_class,
                (1,),
                "float32",  # This is the only type allowed by cuBLAS for AMAX.
                self.device_id,
                stream_holder,
                verify_strides=False,
            )
            self.mm_desc_ifc.amax_d_pointer = self.aux_outputs["result_amax"].data_ptr

        if self.options.block_scaling and self.d_dtype_width == 8:
            self.aux_outputs["d_out_scale"] = utils.create_empty_tensor(
                self.result_class,
                self.mm_traits.d_layout.shape[0] * self.mm_traits.d_layout.shape[1] // 32,  # type: ignore
                "uint8",
                self.device_id,
                stream_holder,
                verify_strides=False,
            )
            self.mm_desc_ifc.d_out_scale_pointer = self.aux_outputs["d_out_scale"].data_ptr

        a, b = self.operands[0], self.operands[1]
        raw_workspace_ptr_device = utils.get_ptr_from_memory_pointer(self.workspace_device)
        if log_info:
            self.logger.info("Starting distributed matrix multiplication...")
            self.logger.info(f"{self.call_prologue}")
        with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            cublasMp.set_stream(self.handle, stream_holder.ptr)

            nullptr = 0
            cublasMp.matmul(
                self.handle,
                self.mm_desc,
                self.mm_traits.M,
                self.mm_traits.N,
                self.mm_traits.K,
                self.alpha.ctypes.data,
                a.data_ptr,
                1,
                1,
                self.matrix_descriptors[0],
                b.data_ptr,
                1,
                1,
                self.matrix_descriptors[1],
                self.beta.ctypes.data,
                nullptr if self.num_operands == 2 else self.operands[2].data_ptr,
                1,
                1,
                self.matrix_descriptors[2],
                result.data_ptr,
                1,
                1,
                self.matrix_descriptors[3],
                raw_workspace_ptr_device,
                self.workspace_size_device,
                self.workspace_host.ctypes.data if self.workspace_size_host > 0 else nullptr,  # type: ignore
                self.workspace_size_host,
            )

        if log_info and elapsed.data is not None:
            self.logger.info(f"The distributed matrix multiplication calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if requested.
        if release_workspace:
            self._release_workspace_memory_perhaps(True)

        # Return the result and auxiliary outputs, if present.
        all_outputs = self.epilog_outputs | self.aux_outputs
        if self.memory_space == "cpu":
            if self.inplace:
                # Overwrite operand C.
                assert self.cpu_c_ref is not None, "Internal error."
                self.cpu_c_ref.copy_(result, stream_holder=stream_holder)
                out = self.cpu_c_ref.tensor
            else:
                out = result.to("cpu", stream_holder=stream_holder).tensor
            # Copy auxiliary output to CPU.
            aux = {name: all_outputs[name].to("cpu", stream_holder=stream_holder).tensor for name in all_outputs}
        else:
            out = result.tensor
            # Return the unwrapped epilog output tensor(s).
            aux = {name: all_outputs[name].tensor for name in all_outputs}

        self.aux_outputs = {}
        self.epilog_outputs = {}
        self._reset_workspace_allocation_tracking()

        if aux:
            return out, aux

        return out

    def free(self):
        """Free Matmul resources.

        It is recommended that the :class:`Matmul` object be used within a context, but if
        it is not possible then this method must be called explicitly to ensure that the
        matrix multiplication resources (especially internal library objects) are properly
        cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the
            # computation.
            if self.last_compute_event is not None:
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            self._free_plan_resources()

            with utils.device_ctx(self.device_id):
                # Destroy matmul descriptor.
                if self.mm_desc is not None:
                    cublasMp.matmul_descriptor_destroy(self.mm_desc)
                    self.mm_desc = None

                # NOTE: cuBLASMp grids are stored in the global cache and destroyed
                # when the cache is cleared (this just clears the references from
                # this object).
                self.lib_process_grids = []

                # Destroy cuBLASMp library handle.
                if self.handle is not None:
                    cublasMp.destroy(self.handle)
                    self.handle = None

            _keep = {"logger", "valid_state"}
            for attr in list(vars(self)):
                if attr not in _keep:
                    setattr(self, attr, None)

        except Exception as e:
            self.logger.critical("Internal error: only part of the Matmul object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The Matmul object's resources have been released.")


@utils.docstring_decorator(SHARED_MM_DOCUMENTATION, skip_missing=False)
def matmul(
    a,
    b,
    /,
    c=None,
    *,
    distributions: Sequence[Distribution],
    alpha=None,
    beta=None,
    epilog=None,
    epilog_inputs=None,
    qualifiers=None,
    quantization_scales=None,
    options=None,
    preferences=None,
    stream: utils.AnyStream | int | None = None,
):
    """
    Perform the specified distributed matrix multiplication computation
    :math:`F(\\alpha a @ b + \\beta c)`, where :math:`F` is the epilog. This function-form
    is a wrapper around the stateful :class:`Matmul` object APIs and is meant for *single*
    use (the user needs to perform just one matrix multiplication, for example), in which
    case there is no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`MatmulOptions` or by setting the
    appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        distributions: {distributions}

        alpha: {alpha}

        beta: {beta}

        epilog: {epilog}

        epilog_inputs: {epilog_inputs}

        qualifiers: {qualifiers}

        quantization_scales: {quantization_scales}

        options: {options}

        preferences: {preferences}

        stream: {stream}

    Returns:
        {result}

    Semantics:
        {semantics}

    Narrow-precision support:
        {narrow_precision}

    .. seealso::
        :class:`Matmul`, :class:`MatmulOptions`, :class:`MatmulEpilog`,
        :class:`MatmulPlanPreferences`

    Examples:

        >>> import cupy as cp
        >>> import nvmath.distributed
        >>> from nvmath.distributed.distribution import Slab

        Get process group used to initialize nvmath.distributed (for information on
        initializing ``nvmath.distributed``, you can refer to the documentation or to the
        Matmul examples in `nvmath/examples/distributed/linalg/advanced
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_):

        >>> process_group = nvmath.distributed.get_context().process_group

        Get my process rank:

        >>> rank = process_group.rank

        Create three float32 ndarrays on the GPU:

        >>> M, N, K = 128, 64, 256
        >>> a_shape = Slab.X.shape(rank, (M, K))
        >>> b_shape = Slab.Y.shape(rank, (K, N))
        >>> c_shape = Slab.X.shape(rank, (M, N))
        >>> device_id = nvmath.distributed.get_context().device_id
        >>> with cp.cuda.Device(device_id):
        ...     a = cp.asfortranarray(cp.random.rand(*a_shape, dtype=cp.float32))
        ...     b = cp.asfortranarray(cp.random.rand(*b_shape, dtype=cp.float32))
        ...     c = cp.asfortranarray(cp.random.rand(*c_shape, dtype=cp.float32))

        Perform the operation :math:`\\alpha A @ B + \\beta C` using :func:`matmul`. The
        result `r` is also a CuPy float32 ndarray:

        >>> distributions = [Slab.X, Slab.Y, Slab.X]
        >>> r = nvmath.distributed.linalg.advanced.matmul(
        ...     a, b, c, alpha=1.23, beta=0.74, distributions=distributions
        ... )

        Options can be provided to customize the operation:

        >>> compute_type = (
        ...     nvmath.distributed.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32
        ... )
        >>> o = nvmath.distributed.linalg.advanced.MatmulOptions(compute_type=compute_type)
        >>> r = nvmath.distributed.linalg.advanced.matmul(
        ...     a, b, distributions=distributions, options=o
        ... )

        See `MatmulOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly
        provided to the Matmul operation. This can be done if the operands are computed on a
        different stream, for example:

        >>> with cp.cuda.Device(device_id):
        ...     s = cp.cuda.Stream()
        ...     with s:
        ...         a = cp.asfortranarray(cp.random.rand(*a_shape))
        ...         b = cp.asfortranarray(cp.random.rand(*b_shape))
        >>> r = nvmath.distributed.linalg.advanced.matmul(
        ...     a, b, distributions=distributions, stream=s
        ... )

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create NumPy ndarrays on the CPU.

        >>> import numpy as np
        >>> a = np.asfortranarray(np.random.rand(*a_shape))
        >>> b = np.asfortranarray(np.random.rand(*b_shape))

        Provide the NumPy ndarrays to :func:`matmul`, with the result also being a NumPy
        ndarray:

        >>> r = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions)

    Notes:
        - This function is a convenience wrapper around :class:`Matmul` and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/distributed/examples/linalg/advanced/matmul
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul>`_
    directory.
    """
    preferences = utils.check_or_create_options(
        _configuration.MatmulPlanPreferences, preferences, "Matrix multiplication plan preferences"
    )

    with Matmul(
        a,
        b,
        c=c,
        distributions=distributions,
        alpha=alpha,
        beta=beta,
        qualifiers=qualifiers,
        options=options,
        stream=stream,
        quantization_scales=quantization_scales,
    ) as mm:
        mm.plan(preferences=preferences, epilog=epilog, epilog_inputs=epilog_inputs, stream=stream)

        r = mm.execute(stream=stream)

    return r


class CublasMpMemoryManager(memory.BaseCUDAMemoryManager):
    """
    cuBLASMp memory allocator. Calls cublasMp.malloc()/cublasMp.free()
    which itself uses ncclMemAlloc/ncclMemFree, and (de)registers the
    buffers with NCCL subcommunicators associated with the given cuBLASMp
    process grids.
    """

    class SymmetricMemoryPointer(memory.MemoryPointer):
        def __init__(self, grids, ptr, size):
            super().__init__(ptr, size, finalizer=None)
            self.grids = grids

        def free(self):
            """**This is a collective call**"""
            for grid in self.grids[1:][::-1]:
                cublasMp.buffer_deregister(grid, self.device_ptr)
            cublasMp.free(self.grids[0], self.device_ptr)

    def __init__(self, device_id: int, grids: list[int], logger: logging.Logger):
        """
        Create CublasMpMemoryManager.

        Args:
            device_id: The ID (int) of the device on which memory is to be allocated.

            grids: List of pointers to cuBLASMp process grids. Memory allocated
                with this manager will be registered with the NCCL subcommunicators
                associated with these grids.

           logger (logging.Logger): Python Logger object.
        """
        self.device_id = device_id
        self.grids = grids
        self.logger = logger

    def memalloc(self, size):
        """**This is a collective call**"""
        assert self.grids, "Internal error"
        # Note: this is a blocking call (it calls ncclMemAlloc which doesn't take a stream).
        ptr = cublasMp.malloc(self.grids[0], size)
        for grid in self.grids[1:]:
            cublasMp.buffer_register(grid, ptr, size)

        self.logger.debug(
            f"CublasMpMemoryManager (allocate memory): size = {size}, pointer = {ptr}, device_id = {self.device_id}"
        )

        return CublasMpMemoryManager.SymmetricMemoryPointer(self.grids, ptr, size)


class cuBLASMpProcessGridCache:
    def __init__(self):
        self.cache = {}
        self.cache_alt = {}
        self.device_id = None
        import threading

        self.lock = threading.Lock()

    def get_library_process_grid(self, process_grid, device_id, nccl_comm, from_alt=False):
        """**This is a collective call**. Caller must make sure to set device context."""
        with self.lock:
            if self.device_id is None:
                self.device_id = device_id
            else:
                assert self.device_id == device_id
            cache = self.cache if not from_alt else self.cache_alt
            if process_grid not in cache:
                process_grid_cpp = cublasMp.grid_create(
                    process_grid.shape[0],
                    process_grid.shape[1],
                    process_grid.layout,
                    nccl_comm.ptr,
                )
                cache[process_grid] = process_grid_cpp
                return process_grid_cpp
            return cache[process_grid]

    def clear(self):
        """This is a collective call."""
        with self.lock:
            if len(self.cache) == 0 and len(self.cache_alt) == 0:
                return
            with utils.device_ctx(self.device_id):
                for cache in (self.cache, self.cache_alt):
                    for grid in cache.values():
                        cublasMp.grid_destroy(grid)
                    cache.clear()


_grid_cache = cuBLASMpProcessGridCache()
