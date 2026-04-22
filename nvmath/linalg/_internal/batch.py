# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # allows typehint of class methods to return the self class

import math
from collections.abc import Sequence
from dataclasses import dataclass

from nvmath._internal.layout import is_overlapping_layout
from nvmath.linalg._internal.utils import axis_order_in_memory, calculate_strides, check_batch_tileable


@dataclass(slots=True, frozen=True)
class BatchTraits:
    """Represents the traits of a batched data.

    A BatchTraits is valid if the non-batch dimensions are contiguous and dense (the data
    span the entire extent of the tensor and there are no gaps between adjacent elements).
    The batch dimensions may be optionally overlapping.

    Attributes:
        shape: The dimensions of the batch. An empty tuple `()` indicates no batching.
        strides: The memory strides for each dimension of the batch.
        overlap_allowed: Whether the batch dimensions are allowed to overlap in memory.

    """

    shape: Sequence[int]
    strides: Sequence[int]
    overlap_allowed: bool = False

    @classmethod
    def from_full_shape_and_strides(
        cls,
        shape: Sequence[int],
        strides: Sequence[int],
        num_trailing_dims: int,
        overlap_allowed: bool = False,
    ) -> BatchTraits:
        """Create BatchTraits from the shape and strides of a tensor.

        Args:
            shape: The full shape of the tensor including both batch and non-batch
                dimensions.
            strides: The memory strides for each dimension of the full tensor.
            num_trailing_dims: The number of trailing (non-batch) dimensions to exclude from
                the batch. For example, for a batched matrix with shape (B1, B2, M, N),
                num_trailing_dims=2 would extract (B1, B2) as batch dimensions.
            overlap_allowed: Whether the batch dimensions are allowed to overlap in memory.
                If False, raises ValueError if overlapping layout is detected. Defaults to
                False.

        Returns:
            BatchTraits with the extracted batch shape and strides.

        Raises:
            ValueError: If the batch layout is not tileable (cannot be efficiently
                processed).
            ValueError: If overlap_allowed is False and the batch layout has overlapping
                memory.

        Example:
            >>> # Extract batch info from a tensor of shape (3, 4, 5, 6)
            >>> # treating last 2 dimensions as non-batch
            >>> batch = BatchTraits.from_full_shape_and_strides(
            ...     shape=(3, 4, 5, 6),
            ...     strides=(120, 30, 6, 1),  # Row-major strides
            ...     num_trailing_dims=2,  # Last 2 dims are non-batch
            ... )
            >>> batch.shape
            (3, 4)
            >>> batch.strides
            (120, 30)
        """
        assert len(shape) == len(strides), (shape, strides)
        num_dims = len(shape)
        leading_shape = shape[: num_dims - num_trailing_dims]
        leading_strides = strides[: num_dims - num_trailing_dims]
        # Check for valid batching
        # The samples in the batch must be tileable.
        if leading_shape and not check_batch_tileable(leading_shape, leading_strides):
            message = (
                f"The batch layout for shape = {leading_shape} and strides = "
                f"{leading_strides} is currently not supported because it is not tileable."
            )
            raise ValueError(message)

        if not overlap_allowed and is_overlapping_layout(leading_shape, leading_strides):
            raise ValueError(
                "Only non-overlapping tensors are valid for batching. "
                f"Shape {leading_shape} with strides {leading_strides} is not valid for batching."
            )
        # It's OK if we prune off the information about the trailing dimensions because the
        # strides should already contain information about them.
        return BatchTraits(
            shape=leading_shape,
            strides=leading_strides,
            overlap_allowed=overlap_allowed,
        )

    @classmethod
    def from_batch_shape_and_size(
        cls,
        shape: Sequence[int],
        axis_order: Sequence[int],
        batch_stride: int,
    ) -> BatchTraits:
        """Create BatchTraits from a batch shape and the size of one batch sample.

        This method constructs BatchTraits by calculating the appropriate strides for a new
        tensor with the given batch `shape`. The strides are computed based on the number of
        elements of one batch sample and the desired layout order specified via
        `axis_order`.

        Args:
            shape: The shape of the batch dimensions
            axis_order: The order of batch axes in memory
            batch_stride: Number of elements of one batch sample (e.g., M*N
                for an MxN matrix)

        Returns:
            BatchTraits with the specified batch shape and computed strides.

        Example:
            >>> # Creating batch info for tensor of shape (3, 4, 5, 5)
            >>> batch = BatchTraits.from_batch_shape_and_size(
            ...     shape=(3, 4),  # 2D batch
            ...     axis_order=(1, 0),  # ROW order,inner dimension (axis 1) comes first
            ...     batch_stride=25,  # Each 5x5 matrix = 25 elements
            ... )
            >>> batch.shape
            (3, 4)
            >>> batch.strides
            (100, 25)  # Jump 100 for outer dim, 25 for inner dim
        """
        strides = calculate_strides(
            shape,
            axis_order,
            batch_stride,
        )
        return BatchTraits.from_full_shape_and_strides(
            shape=shape,
            strides=strides,
            num_trailing_dims=0,
            overlap_allowed=False,
        )

    @property
    def count(self) -> int:
        """The total number of elements in the batch. Returns 0 if non-batched."""
        return 0 if not self.shape else math.prod(self.shape)

    @property
    def stride(self) -> int:
        """The stride between elements the batch. Returns 0 if non-batched."""
        return 0 if not self.shape else min(self.strides)

    @property
    def axis_order(self) -> tuple[int, ...]:
        """The axis order of the batch. Returns an empty tuple if non-batched."""
        return axis_order_in_memory(self.strides)

    def __eq__(self, other) -> bool:
        """Return whether two BatchTraits are equal.

        Two BatchTraits are equal only if both the shape and strides of the batches are
        equal.
        """
        if not isinstance(other, BatchTraits):
            raise TypeError("Unsupported operand type(s) for ==.")
        return self.shape == other.shape and self.strides == other.strides

    def __mul__(self, other: BatchTraits | tuple[Sequence[int], Sequence[int]]) -> tuple[Sequence[int], Sequence[int]]:
        """Return the combined shape of two BatchTraits.

        Currently two BatchTraits are only combinable if they have the same shape and axis
        order or if one is an empty batch.

        Returns
        -------
        shape: The combined shape of the two BatchTraits.
        axis_order: The combined axis order of the two BatchTraits.

        Raises
        ------
        ValueError: if the dimensions and strides are incompatible.
        """
        if isinstance(other, BatchTraits):
            shape, axis_order = other.shape, other.axis_order
        else:
            shape, axis_order = other  # type: ignore[assignment]
        if not self.shape or self.shape == shape:
            new_shape = shape
        elif not shape:
            new_shape = self.shape
        else:
            msg = (
                f"Batch dimensions {self.shape} and {shape} are not compatible. "
                "Dimensions must be the same OR one input must be non-batched."
            )
            raise ValueError(msg)
        if not self.axis_order or self.axis_order == axis_order:
            new_axis_order = axis_order
        elif not axis_order:
            new_axis_order = self.axis_order
        else:
            msg = (
                f"Batch order {self.axis_order} and {axis_order} are not compatible. "
                "Batch orders must be the same OR one input must be non-batched."
            )
            raise ValueError(msg)
        return new_shape, new_axis_order

    def __str__(self) -> str:
        return f"shape {self.shape} and strides {self.strides}"
