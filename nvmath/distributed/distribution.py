from __future__ import annotations  # allows typehint of class methods to return the self class

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast, TypeAlias


from nvmath.bindings import cufftMp  # type: ignore
from nvmath.bindings import cublasMp  # type: ignore

import nvmath.distributed as dist

__all__ = ["ProcessGrid", "Distribution", "Slab", "Box", "BlockCyclic", "BlockNonCyclic"]


class ProcessGrid:
    """
    N-dimensional grid of processes used by some distributions like the PBLAS block-cyclic
    distribution.

    Example 2D process grid for 4 processes, with processes arranged in column-major order::

        ---------
        | 0 | 2 |
        ---------
        | 1 | 3 |
        ---------
    """

    Layout: TypeAlias = cublasMp.GridLayout

    def __init__(
        self,
        *,
        shape: Sequence[int] | None = None,
        layout: ProcessGrid.Layout | None = None,
        process_array=None,
    ):
        """
        Create a new ProcessGrid object.

        Args:
            shape: Shape of the process grid.

            layout: Layout of the process grid (column-major or row-major). This is optional
                for 1D grid or when a custom grid is provided.

            process_array: optional ndarray specifying custom arrangement of processes.
        """
        self._nranks = _get_communicator().Get_size()

        if process_array is None:
            if shape is None:
                raise ValueError("shape must be provided when process_array=None")
            self._shape = tuple(shape)
            if layout is None:
                if self._is_1d_distribution():
                    layout = ProcessGrid.Layout.ROW_MAJOR  # layout doesn't matter in this case.
                else:
                    raise ValueError("layout must be provided when process_array=None and partitioning on multiple dimensions")
            if not isinstance(layout, ProcessGrid.Layout):
                raise TypeError(f"layout must be of type ProcessGrid.Layout, got {layout}")
            self._layout = layout
            self._process_array = None
        else:
            self._shape = tuple(process_array.shape)
            if shape is not None and self._shape != tuple(shape):
                raise ValueError(f"shape {shape} and process_array.shape ({process_array.shape}) don't match")
            # TODO: Can set layout to COL_MAJOR or ROW_MAJOR automatically if the
            # process_array matches.
            if layout is not None:
                raise NotImplementedError
            self._layout = None
            self._process_array = process_array

        if math.prod(self._shape) != self._nranks:
            raise ValueError(
                f"Number of grid elements ({math.prod(self._shape)}) must equal the number of processes ({self._nranks})"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of process grid."""
        return self._shape

    @property
    def layout(self) -> ProcessGrid.Layout | None:
        """Layout of process grid if row-major or column-major, otherwise None."""
        return self._layout

    @property
    def process_array(self):
        return self._process_array

    def __str__(self):
        return f"ProcessGrid(shape={self._shape}, layout={self._layout.name}, process_array={self._process_array})"

    def __hash__(self):
        # NOTE: The layout isn't considered for the hash, to allow process grids
        # partitioned on a single dimension with the same shape but different layout
        # to be the same dictionary key.
        return hash(self._shape)

    def __eq__(self, other):
        if self._process_array is None and other._process_array is None:
            if self._shape == other._shape:
                return self._is_1d_distribution() or self._layout == other._layout
            return False
        raise NotImplementedError

    def _is_1d_distribution(self) -> bool:
        """True if process grid partitions on a single dimension."""
        return self._nranks in self._shape

    def _is_row_wise(self) -> bool:
        """True if 2D process grid partitioned only on rows."""
        return self._shape == (self._nranks, 1)

    def _is_col_wise(self) -> bool:
        """True if 2D process grid partitioned only on columns."""
        return self._shape == (1, self._nranks)


class BindDistributionError(Exception):
    pass


class ConvertDistributionError(Exception):
    """Errors converting a distribution instance to another distribution type"""

    pass


class Distribution(ABC):
    """Specifies how a tensor is distributed across processes."""

    def __init__(self):
        self._bound = False

    @property
    @abstractmethod
    def ndim(self) -> int | None:
        """The number of dimensions of a distributed tensor for which this distribution
        applies; None if it doesn't apply to any specific number of dimensions."""
        raise NotImplementedError

    @abstractmethod
    def shape(self, rank: int, global_shape: Sequence[int] | None = None) -> tuple[int, ...]:
        """Get the local shape of data on the given rank according to this distribution.

        Args:
            rank: the process rank for which to calculate the local shape.

            global_shape: Global shape of data. Required if the distribution is
                not bound to a global shape, otherwise not required.
        """
        raise NotImplementedError

    @abstractmethod
    def to(
        self,
        cls: type[Distribution],
        /,
        *,
        ndim: int | None = None,
        copy: bool = False,
    ) -> Distribution:
        """Convert this distribution object to an equivalent distribution of the given type.

        Args:
            cls: the target distribution type.

            ndim: dimensionality of the target distribution. Must be compatible with the
                dimensionality of the source distribution. This may be required if the
                source distribution doesn't have associated dimensionality.

            copy: Returns a copy if the source and target type are the same.

        Raises:
            ConvertDistributionError: if the conversion is not possible.
        """
        raise NotImplementedError

    def _to_checks(self, cls: type[Distribution], ndim: int | None) -> None:
        if cls is Distribution or not issubclass(cls, Distribution):
            raise ValueError(f"{cls} is not a valid distribution")
        if ndim is not None and self.ndim is not None and ndim != self.ndim:
            raise ValueError(f"ndim argument ({ndim}) doesn't match this distribution's dimensionality ({self.ndim})")

    @abstractmethod
    def _bind(
        self,
        global_shape: Sequence[int],
        *,
        shape: Sequence[int] | None = None,
    ) -> Distribution:
        """Binds this distribution object to a global shape, which determines how a
        distributed tensor with that shape must be partitioned among processes (the local
        shape on each process). You can also provide the local shape on this process to
        check if it fits the distribution (the function will raise an exception if not).
        **The exception may be raised on some ranks but not others (it's up to the caller
        to handle this)**.

        Args:
            global_shape: global shape of the data.

            shape: shape of the data on this process.

        Returns:
            self

        Raises:
            BindDistributionError: if distribution is already bound or local shape doesn't
            fit the distribution.
        """
        raise NotImplementedError

    def _binding_str(self):
        return f"[bound: global_shape={self._data_global_shape}, shape={self._data_shape}]" if self._bound else ""

    def copy(self) -> Distribution:
        """This is a common implementation for those distributions that only require
        a shallow copy."""
        return copy.copy(self)

    def _local_shape_checks(self, rank: int, nranks: int, global_shape: Sequence[int] | None = None):
        if not isinstance(rank, int):
            raise ValueError(f"rank must be an integer, got {rank}")

        if rank < 0 or rank > nranks - 1:
            raise ValueError(f"This is not a valid process rank: got rank={rank} with nranks={nranks}")

        if global_shape is None and not self._bound:
            raise RuntimeError("This distribution is unbound: please specify a global shape")

        if global_shape is not None and self._bound and tuple(global_shape) != self._data_global_shape:  # type: ignore
            raise ValueError(
                "This distribution is already bound to a different global shape: provided "
                f"{global_shape}, bound to {self._data_global_shape}"  # type: ignore
            )


class Slab(Distribution):
    """
    Slab distribution

    Data is partitioned across processes on a single axis, such that:

    - The shape of the slab on the first s_p % P processes is
      (s_0, ..., s_p // P + 1, ..., s_{n-1})
    - The shape of the slab on the remaining processes is (s_0, ..., s_p // P, ..., s_{n-1})
    - Process 0 owns the first slab according to the global index order, process 1 owns
      the second slab and so on.

    where:

    - s_i is the size of dimension i of the global array
    - p is the partition dimension
    - n is the number of dimensions of the array
    - P is the number of processes
    """

    X: Slab
    """Slab distribution on axis 0."""

    Y: Slab
    """Slab distribution on axis 1."""

    Z: Slab
    """Slab distribution on axis 2."""

    def __init__(
        self,
        partition_dim: int,
        ndim: int | None = None,
    ):
        super().__init__()
        if not isinstance(partition_dim, int) or partition_dim < 0:
            raise ValueError(f"partition_dim must be integer >= 0, got {partition_dim}")

        if ndim is not None:
            if not isinstance(ndim, int) or ndim < 1:
                raise ValueError(f"ndim must be integer >= 1, got {ndim}")
            if partition_dim >= ndim:
                raise ValueError("partition_dim must be < ndim")

        self._partition_dim = partition_dim
        self._ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, Slab):
            return False
        if self._partition_dim != other._partition_dim:
            return False
        if self._ndim is None or other._ndim is None:
            return True
        return self._ndim == other._ndim

    def __hash__(self):
        return self._partition_dim

    def __str__(self):
        return f"Slab(partition_dim={self._partition_dim}, ndim={self._ndim})" + self._binding_str()

    @property
    def name(self) -> str:
        match self._partition_dim:
            case 0:
                return "Slab.X"
            case 1:
                return "Slab.Y"
            case 2:
                return "Slab.Z"
            case _:
                return "Slab"

    @property
    def partition_dim(self) -> int:
        """Slab partition dimension"""
        return self._partition_dim

    @property
    def ndim(self) -> int | None:
        return self._ndim

    def shape(self, rank: int, global_shape: Sequence[int] | None = None) -> tuple[int, ...]:
        comm = _get_communicator()
        n = comm.Get_size()
        self._local_shape_checks(rank, n, global_shape)
        if global_shape is None:
            global_shape = self._data_global_shape

        S = global_shape[self._partition_dim]
        partition_dim_local_size = S // n + bool(rank < S % n)
        slab_shape = list(global_shape)
        slab_shape[self._partition_dim] = partition_dim_local_size
        return tuple(slab_shape)

    @property
    def _cufftmp_value(self):
        if self._partition_dim not in (0, 1):
            raise TypeError(f"Unsupported distribution {self} for cuFFTMp: partition dimension must be X or Y")
        return cufftMp.XtSubFormat.FORMAT_INPLACE if self._partition_dim == 0 else cufftMp.XtSubFormat.FORMAT_INPLACE_SHUFFLED

    def _bind(self, global_shape, *, shape=None) -> Slab:
        if self._bound:
            raise BindDistributionError(f"{self} is already bound")

        if self._ndim is not None and len(global_shape) != self._ndim:
            raise ValueError(f"The given shape doesn't have the same dimensionality as this {self} distribution")

        if self._partition_dim >= len(global_shape):
            raise ValueError("partition_dim must be < ndim")

        comm = _get_communicator()
        rank = comm.Get_rank()
        slab_shape = self.shape(rank, global_shape)

        if shape is not None and tuple(shape) != slab_shape:
            raise BindDistributionError(
                f"The given shapes (global_shape={global_shape}, shape={shape}) don't fit distribution {str(self)}"
            )

        self._ndim = len(global_shape)
        self._data_global_shape = tuple(global_shape)
        self._data_shape = slab_shape
        self._bound = True
        return self

    def to(self, cls, /, *, ndim=None, copy=False):
        super()._to_checks(cls, ndim)
        nranks = _get_communicator().Get_size()

        if ndim is None:
            ndim = self._ndim
        elif self._partition_dim >= ndim:
            raise ValueError(f"ndim ({ndim}) must be greater than the partition dimension ({self._partition_dim})")

        if cls is Slab:
            if copy or (ndim is not None and self._ndim is None):
                d = cast(Slab, self.copy())
                d._ndim = ndim
                return d
            return self
        elif issubclass(cls, BlockCyclic):
            if ndim is None:
                raise ConvertDistributionError("Can't convert Slab distribution to BlockCyclic: unknown dimensionality")

            process_grid_shape = tuple(1 if x != self._partition_dim else nranks for x in range(ndim))
            # layout doesn't matter when partitioning on a single axis.
            process_grid = ProcessGrid(shape=process_grid_shape, layout=ProcessGrid.Layout.ROW_MAJOR)
            if not self._bound:
                return BlockNonCyclic(process_grid)
            else:
                b = BlockNonCyclic(process_grid)
                return b._bind(self._data_global_shape, shape=self._data_shape)
        elif cls is Box:
            raise NotImplementedError


# Define alternate forms for the user to specify Slab on X, Y or Z.
# NOTE: dimensionality is left unspecified when using these (but will be set when the
# distribution is bound to data).
Slab.X = Slab(0)
Slab.Y = Slab(1)
Slab.Z = Slab(2)


class Box(Distribution):
    """Box distribution"""

    def __init__(
        self,
        lower: Sequence[int],
        upper: Sequence[int],
    ):
        super().__init__()
        if len(lower) != len(upper):
            raise ValueError("lower and upper coordinates must have the same dimensionality")
        for coords in (lower, upper):
            if not all(isinstance(x, int) for x in coords):
                raise ValueError("lower and upper coordinates must be integer")
        if not all(upper[i] > lower[i] for i in range(len(upper))):
            raise ValueError(
                f"The upper coordinates must be larger than the lower coordinates, but got lower={lower} upper={upper}"
            )
        self._lower = tuple(lower)
        self._upper = tuple(upper)

    @property
    def lower(self) -> tuple[int, ...]:
        """Box lower coordinates"""
        return self._lower

    @property
    def upper(self) -> tuple[int, ...]:
        """Box upper coordinates"""
        return self._upper

    @property
    def ndim(self) -> int:
        return len(self._lower)

    def shape(self, rank: int, global_shape: Sequence[int] | None = None) -> tuple[int, ...]:
        comm = _get_communicator()
        if rank != comm.Get_rank():
            raise RuntimeError("Can't calculate local shape of peer process with Box distribution")
        nranks = comm.Get_size()
        self._local_shape_checks(rank, nranks, global_shape)
        return tuple(self._upper[i] - self._lower[i] for i in range(self.ndim))

    def __str__(self):
        return f"Box(lower={self._lower}, upper={self._upper})" + self._binding_str()

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return self._lower == other._lower and self._upper == other._upper

    def __hash__(self):
        return hash((self._lower, self._upper))

    def __iter__(self):
        # To allow unpacking
        yield self._lower
        yield self._upper

    def __getitem__(self, index):
        if index not in (0, 1):
            return IndexError(f"Index must be 0 or 1, got {index}")
        return self._lower if index == 0 else self._upper

    def _bind(self, global_shape, *, shape=None) -> Distribution:
        if self._bound:
            raise BindDistributionError(f"{self} is already bound")
        my_shape = tuple(self._upper[i] - self._lower[i] for i in range(self.ndim))
        if shape is not None and tuple(shape) != my_shape:
            raise BindDistributionError(
                f"The given shapes don't fit this Box distribution: {global_shape} "
                f"and {shape}, lower={self._lower} upper={self._upper}"
            )
        self._data_global_shape = tuple(global_shape)
        self._data_shape = my_shape
        self._bound = True
        return self

    def to(self, cls, /, *, ndim=None, copy=False):
        super()._to_checks(cls, ndim)
        if cls is Box:
            return self.copy() if copy else self
        raise NotImplementedError


class BlockCyclic(Distribution):
    """Block-cyclic distribution"""

    def __init__(
        self,
        process_grid: ProcessGrid,
        block_sizes: Sequence[int],
        *,
        first_process: Sequence[int] | None = None,
    ):
        super().__init__()
        if block_sizes is None or not all(isinstance(x, int) for x in block_sizes):
            raise ValueError(f"Must provide a sequence of integer block sizes, got {block_sizes}")
        if len(block_sizes) != len(process_grid.shape):
            raise ValueError(
                f"Number of block sizes ({len(block_sizes)}) doesn't match dimensionality ({len(process_grid.shape)})"
            )
        self._process_grid = process_grid
        self._block_sizes = tuple(block_sizes)
        if first_process is None:
            self._first_process = (0,) * self.ndim
        else:
            if not all(isinstance(x, int) for x in first_process):
                raise ValueError(f"first_process must be a sequence of integer coordinates, got {first_process}")
            for i, x in enumerate(first_process):
                if x < 0 or x >= process_grid.shape[i]:
                    raise ValueError(
                        f"first_process {first_process} is not a valid index into the process grid of "
                        f"shape {process_grid.shape}"
                    )
            self._first_process = tuple(first_process)

    @property
    def process_grid(self) -> ProcessGrid:
        """The process grid of this BlockCyclic distribution"""
        return self._process_grid

    @property
    def ndim(self) -> int:
        return len(self._process_grid.shape)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """The block sizes of this BlockCyclic distribution"""
        return self._block_sizes

    @property
    def first_process(self) -> tuple[int, ...]:
        """Index in the process grid of the process who owns the first block of the
        distributed tensor."""
        return self._first_process

    def __str__(self):
        return (
            f"{self.__class__.__name__}(process_grid={self._process_grid}, block_sizes={self._block_sizes})"
            + self._binding_str()
        )

    def __eq__(self, other):
        if not isinstance(other, BlockCyclic):
            return False
        return (
            self._process_grid == other._process_grid
            and self._block_sizes == other._block_sizes
            and self._first_process == other._first_process
        )

    def _is_1d_distribution(self) -> bool:
        """True if process grid partitions on a single dimension."""
        return self._process_grid._is_1d_distribution()

    def _is_row_wise(self) -> bool:
        """True if 2D process grid partitioned only on rows."""
        return self._process_grid._is_row_wise()

    def _is_col_wise(self) -> bool:
        """True if 2D process grid partitioned only on columns."""
        return self._process_grid._is_col_wise()

    def shape(self, rank: int, global_shape: Sequence[int] | None = None) -> tuple[int, ...]:
        self._local_shape_checks(rank, self._process_grid._nranks, global_shape)
        if global_shape is None:
            global_shape = self._data_global_shape
        return self._calc_local_shape(rank, self._block_sizes, global_shape)

    def _calc_local_shape(self, rank, block_sizes, global_shape):
        nprow, npcol = self._process_grid._shape
        layout = self._process_grid._layout
        if layout is not None:
            myprow = rank % nprow if layout == ProcessGrid.Layout.COL_MAJOR else rank // npcol
            mypcol = rank // nprow if layout == ProcessGrid.Layout.COL_MAJOR else rank % npcol
            index = (myprow, mypcol)
        else:
            raise NotImplementedError
        nrows = cublasMp.numroc(global_shape[0], block_sizes[0], index[0], self._first_process[0], nprow)
        ncols = cublasMp.numroc(global_shape[1], block_sizes[1], index[1], self._first_process[1], npcol)
        return (nrows, ncols)

    def _bind(self, global_shape, *, shape=None) -> Distribution:
        if self._bound:
            raise BindDistributionError(f"{self} is already bound")

        if self.ndim != 2:
            raise NotImplementedError

        if len(global_shape) != self.ndim:
            raise ValueError(
                f"Dimensionality of shapes ({len(global_shape)}) doesn't match dimensionality "
                f"of this distribution ({self.ndim})"
            )

        rank = _get_communicator().Get_rank()
        nrows, ncols = self._calc_local_shape(rank, self._block_sizes, global_shape)
        if shape is not None and tuple(shape) != (nrows, ncols):
            raise BindDistributionError(
                f"The local shape {shape} on process {rank} is not the expected one based "
                f"on the global shape {global_shape}, process grid {self._process_grid} and "
                f"block sizes {self._block_sizes}: expected shape is {(nrows, ncols)}"
            )

        self._data_global_shape = global_shape
        self._data_shape = (nrows, ncols)
        self._bound = True
        return self

    def to(self, cls, /, *, ndim=None, copy=False):
        super()._to_checks(cls, ndim)
        nranks = _get_communicator().Get_size()
        if issubclass(cls, BlockCyclic):
            return self.copy() if copy else self
        elif cls is Slab:
            if not self._bound:
                # Without binding, it's a stretch to assume that this is compatible with
                # Slab (the "cyclic" nature means that it's much more likely that it isn't).
                raise ConvertDistributionError(
                    "Unbound BlockCyclic distribution can't be converted to Slab. "
                    "Consider using BlockNonCyclic if there is no cyclic distribution of blocks."
                )

            if not self._is_1d_distribution():
                raise ConvertDistributionError(
                    "Can't convert this block distribution to Slab: partitioning must be on a single dimension"
                )

            # Data must be divisible on partition_dim and the block size must correspond
            # to the slab size.
            partition_dim = self._process_grid.shape.index(nranks)

            if self._data_global_shape[partition_dim] % nranks != 0:
                raise ConvertDistributionError(
                    "Can't convert this distribution to Slab: data doesn't divide evenly on partition dimension."
                    f" Global shape is {self._data_global_shape}, partition dimension is "
                    f"{partition_dim} and number of processes is {nranks}."
                )

            if self._block_sizes[partition_dim] != self._data_global_shape[partition_dim] // nranks:
                raise ConvertDistributionError(
                    "This distribution can't be converted to Slab, because the block size in the "
                    f"partition dimension {partition_dim} is not a factor of the global extent in "
                    f"that dimension {self._data_global_shape[partition_dim]}"
                )
            d = Slab(partition_dim, ndim=self.ndim)
            return d._bind(self._data_global_shape, self._data_shape)
        elif cls is Box:
            raise NotImplementedError


class BlockNonCyclic(BlockCyclic):
    """Block distribution without cycles"""

    def __init__(
        self,
        process_grid: ProcessGrid,
        *,
        first_process: Sequence[int] | None = None,
    ):
        super().__init__(process_grid, (-1,) * len(process_grid.shape), first_process=first_process)

    def __eq__(self, other):
        if isinstance(other, BlockNonCyclic) and (not other._bound or not self._bound):
            # Don't compare block sizes since one of them is not bound.
            return self._process_grid == other._process_grid and self._first_process == other._first_process
        return super().__eq__(other)

    def shape(self, rank: int, global_shape: Sequence[int] | None = None) -> tuple[int, ...]:
        self._local_shape_checks(rank, self._process_grid._nranks, global_shape)
        if global_shape is None:
            global_shape = self._data_global_shape
        if self._bound:
            block_sizes = self._block_sizes
        else:
            block_sizes = self._infer_block_sizes(global_shape)
        return self._calc_local_shape(rank, block_sizes, global_shape)

    def _infer_block_sizes(self, global_shape):
        if all(x % self._process_grid._shape[i] == 0 for i, x in enumerate(global_shape)):
            block_sizes = tuple(x // self._process_grid._shape[i] for i, x in enumerate(global_shape))
        else:
            # The logic to bind this global shape to this distribution isn't implemented yet
            # (which doesn't necessarily mean that it isn't possible to fit the data to this
            # distribution).
            raise NotImplementedError(
                "BlockNonCyclic is currently only implemented for uniform partition sizes. "
                "Use BlockCyclic with explicit block sizes instead."
            )
        return block_sizes

    def _bind(self, global_shape, *, shape=None) -> Distribution:
        if self._bound:
            raise BindDistributionError(f"{self} is already bound")

        assert all(x == -1 for x in self._block_sizes)

        if len(global_shape) != self.ndim:
            raise ValueError(
                f"Dimensionality of shapes ({len(global_shape)}) doesn't match dimensionality "
                f"of this distribution ({self.ndim})"
            )

        # Now we assign block sizes to this distribution based on the shape of the data.
        # NOTE: For dimensions that aren't partitioned, there are multiple blocks sizes that
        # are valid and still fit the Block[Non]Cyclic model. This is because, in addition
        # to block size being the full length L of the dimension, any L//N is also a valid
        # block size (so a single block in that dimension is equivalent to N contiguous
        # blocks in that dimension). The importance of this is that cuBLASMp actually
        # requires configurations with these block sizes, since block sizes have to match
        # across matrices A, B and C but they might be distributed differently.
        # In other words, for distributed matmul with BlockNonCyclic, block sizes have to be
        # inferred jointly with matrices A, B and C, but that is outside the scope of this
        # method.

        block_sizes = self._infer_block_sizes(global_shape)
        if shape is not None and block_sizes != tuple(shape):
            raise BindDistributionError("Data doesn't fit BlockNonCyclic distribution")

        self._block_sizes = block_sizes
        return super()._bind(global_shape, shape=shape)

    def to(self, cls, /, *, ndim=None, copy=False):
        super()._to_checks(cls, ndim)
        nranks = _get_communicator().Get_size()
        if issubclass(cls, BlockCyclic):
            return self.copy() if copy else self
        elif cls is Slab:
            if not self._is_1d_distribution():
                raise ConvertDistributionError(
                    "Can't convert this block distribution to Slab: partitioning must be on a single dimension"
                )

            partition_dim = self._process_grid.shape.index(nranks)
            d = Slab(partition_dim, ndim=self.ndim)
            # For bound=False, we can allow the conversion and let Slab._bind() catch
            # any potential errors later.
            if self._bound:
                if self._data_global_shape[partition_dim] % nranks != 0:
                    raise ConvertDistributionError(
                        "Can't convert this distribution to Slab: data doesn't divide evenly on partition dimension"
                    )
                d._bind(self._data_global_shape, shape=self._data_shape)
            return d
        elif cls is Box:
            raise NotImplementedError


def _get_communicator():
    distributed_ctx = dist.get_context()
    if distributed_ctx is None:
        raise RuntimeError(
            "nvmath.distributed has not been initialized. Refer to "
            "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html#initializing-the-distributed-runtime"
            " for more information."
        )
    return distributed_ctx.communicator
