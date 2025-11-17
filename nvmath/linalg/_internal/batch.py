from collections.abc import Sequence
import math

from dataclasses import dataclass

from nvmath.linalg._internal.utils import check_batch_tileable
from nvmath._internal.layout import is_overlapping_layout


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
    ):
        leading_shape, _ = shape[:-num_trailing_dims], shape[-num_trailing_dims:]
        leading_strides, _ = strides[:-num_trailing_dims], strides[-num_trailing_dims:]
        # Check for valid batching
        # The samples in the batch must be tileable.
        if leading_shape and not check_batch_tileable(leading_shape, leading_strides):
            message = (
                f"The batch layout corresponding to shape = {leading_shape} and strides = "
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
    def from_full_shape_only(
        cls,
        shape: Sequence[int],
        num_trailing_dims: int,
    ):
        """Create a BatchTraits from a full shape assuming that the strides are dense
        contiguous and row-ordered."""
        leading_shape, _ = shape[:-num_trailing_dims], shape[-num_trailing_dims:]
        strides = () if not shape else tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        leading_strides, _ = strides[:-num_trailing_dims], strides[-num_trailing_dims:]
        return BatchTraits(
            shape=leading_shape,
            strides=leading_strides,
            overlap_allowed=False,
        )

    @property
    def count(self):
        """The total number of elements in the batch. Returns 0 if non-batched."""
        return 0 if not self.shape else math.prod(self.shape)

    @property
    def stride(self):
        """The stride between elements the batch. Returns 0 if non-batched."""
        return 0 if not self.shape else min(self.strides)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BatchTraits):
            raise TypeError("Unsupported operand type(s) for ==.")
        return self.shape == other.shape and self.strides == other.strides

    def __mul__(self, other) -> Sequence[int]:
        """Return the combined shape of two BatchTraits.

        Currently two BatchTraits are only combinable if they are equal or if one is an
        empty batch.

        Raises
        ------
        ValueError: if the dimensions and strides are incompatible.
        """
        if not isinstance(other, BatchTraits):
            raise TypeError("Unsupported operand type(s) for *.")

        if self.shape == other.shape:
            return self.shape
        if not self.shape:
            return other.shape
        if not other.shape:
            return self.shape

        msg = (
            f"Batch dimensions {self.shape} and {other.shape} are not compatible."
            "Dimensions must be the same OR one input must be non-batched."
        )
        raise ValueError(msg)

    def __str__(self) -> str:
        return f"batch dimensions with shape {self.shape} and strides {self.strides}"
