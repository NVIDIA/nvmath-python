import abc
import contextlib
import dataclasses
import logging

from logging import Logger
from typing import Literal, ClassVar, Final, TypeVar, Generic
from collections.abc import MutableSequence

from nvmath.internal import utils
from nvmath import memory


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCUDA:
    """
    A data class for providing GPU execution options.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

    .. seealso::
       :class:`ExecutionCPU`
    """

    name: ClassVar[Literal["cuda"]] = "cuda"
    device_id: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCPU:
    """
    A data class for providing CPU execution options.

    Attributes:
        num_threads: The number of CPU threads used to execute the operation.
                     If not specified, defaults to the number of CPU cores available to the
                     process.

    .. seealso::
       :class:`ExecutionCUDA`
    """

    name: ClassVar[Literal["cpu"]] = "cpu"
    num_threads: int | None = None


def copy_operand_perhaps(
    internal_operand: utils.TensorHolder | None,
    operand: utils.TensorHolder,
    stream_holder: utils.StreamHolder | None,
    execution_device_id: int | Literal["cpu"],
    operands_device_id: int | Literal["cpu"],
) -> tuple[utils.TensorHolder, utils.TensorHolder | None]:
    """Private implementation of memory space management for tensor operands.

    The `copy_operand_perhaps` function facilitates transitions of tensor operands between
    different memory spaces, ensuring compatibility with execution requirements. Its role is
    to determine whether a tensor operand needs to be copied to accommodate differing
    execution and operand memory spaces, while preserving the original operand for cases
    requiring in-place operations.

    Args:
        internal_operand: Represents an internal tensor for in-place
            memory operations, or `None` if not applicable.

        operand: Tensor to possibly copied to the execution memory space.

        stream_holder: Manages the CUDA stream for device operations.

        execution_device_id: Specifies the target execution space.

        operands_device_id: Specifies the current operand memory space.

    Returns:
        A tuple containing:
            - The operand copied to the execution space, or the original operand if
              no copy is necessary.
            - The original operand, or `None` if no copy occurred.

    """
    if execution_device_id == operands_device_id:
        return operand, None
    else:
        # Copy the `operand` to memory that matches the exec space
        # and keep the original `operand` to handle `options.inplace=True`
        if internal_operand is None:
            exec_space_copy = operand.to(execution_device_id, stream_holder)
            return exec_space_copy, operand
        else:
            # In-place copy to existing pointer
            internal_operand.copy_(src=operand, stream_holder=stream_holder)
            return internal_operand, operand


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StatefulAPIOptions:
    """A dataclass for providing options to a :class:`StatefulAPI` object.

    Attributes:
        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used
            to draw device memory. If an allocator is not provided, a memory allocator from
            the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for
            PyTorch operands, :func:`cupy.cuda.alloc` otherwise).

        blocking: A flag specifying the behavior of the stream-ordered functions and
            methods. When ``blocking`` is `True`, the stream-ordered methods do not return
            until the operation is complete. When ``blocking`` is ``"auto"``, the methods
            return immediately when the inputs are on the GPU. The stream-ordered methods
            always block when the operands are on the CPU to ensure that the user doesn't
            inadvertently use the result before it becomes available. The default is
            ``"auto"``.

        logger: Python Logger object. The root logger will be used if a
            logger object is not provided.

    .. seealso::
       :class:`StatefulAPI`
    """

    allocator: memory.BaseCUDAMemoryManager | memory.BaseCUDAMemoryManagerAsync | None = None
    blocking: Literal[True, "auto"] = "auto"
    logger: Logger = dataclasses.field(default_factory=logging.getLogger)

    def __post_init__(self):
        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for blocking must be either True or 'auto'.")

        if self.allocator is not None and not isinstance(
            self.allocator, memory.BaseCUDAMemoryManager | memory.BaseCUDAMemoryManagerAsync
        ):
            raise TypeError("The allocator must be an object of type that fulfills the BaseCUDAMemoryManager protocol.")


OptionsPlaceholder = TypeVar("OptionsPlaceholder", bound=StatefulAPIOptions)


class StatefulAPI(contextlib.AbstractContextManager, Generic[OptionsPlaceholder]):
    """A base class for APIs which amortize setup costs across multiple executions.

    StatefulAPIs separate planning (``plan()``) and setup (``__init__()``) actions from
    execution (``_execute()``), so that plans may be reused with different operands. The
    ``reset_operands()`` method allows changing the operands of the API without replanning
    when the input and execution space do not match (the user does not have a reference to
    the execution space buffers). If the execution and input space match, we expect the
    user to be able update the operands by overwriting their buffers in-place.
    """

    # options is declared Final in __init__() because mypy issue #8982 is not fixed yet.
    # options: Final[OptionsPlaceholder]
    # See docstring for StatefulAPIOptions
    _allocator: Final[memory.BaseCUDAMemoryManager | memory.BaseCUDAMemoryManagerAsync | None]
    _blocking: Final[bool]
    _logger: Final[logging.Logger]

    # Metadata related to execution space
    execution: Final[ExecutionCPU | ExecutionCUDA]
    """A class which describes the execution space parameters."""
    _internal_op_package: Final[str]
    """The package of the operands in the execution space."""
    _operands: MutableSequence[utils.TensorHolder]
    """A copy of the operands in execution space."""
    _result_class: Final[type[utils.TensorHolder]]
    """The type of TensorHolder to use for the execution space result."""

    # Metadata about the input/output tensors
    _operands_backup: MutableSequence[utils.TensorHolder | None]
    """A reference to original operands in their input space."""
    _operands_device_id: Final[int | Literal["cpu"]]
    """The device_id of the input space."""
    _operands_package: Final[str]
    """The package of the operands in the input space."""

    _call_prologue: Final[str]
    """Stores a message for logging about blocking behavior"""

    _has_plan: bool
    """True if plan has been called."""

    @property
    def options(self) -> OptionsPlaceholder:
        """The options object used to construct this class."""
        # This is a workaround for mypy issue #8982, where we cannot declare options as
        # Final in the class definition, but we still want it to appear in the docs as an
        # attribute.
        return self._options

    def __init__(
        self,
        operands: MutableSequence[utils.TensorHolder],
        *,
        options: OptionsPlaceholder,
        execution: ExecutionCPU | ExecutionCUDA | None | Literal["cuda", "cpu"] = None,
        stream: utils.AnyStream | int | None = None,
    ) -> None:
        """Copy operands to the execution space and setup options.

        When inheriting from this class, you must create valid operands and options in
        the child class before calling StatefulAPI.__init__( ... ).
        """
        self._options: Final[OptionsPlaceholder] = options
        self._logger = self._options.logger

        self._logger.info("= SPECIFICATION PHASE =")

        operands_device_id = utils.get_operands_device_id(operands)

        match execution, operands_device_id:
            case (None | "cuda", int()):
                execution = ExecutionCUDA(device_id=operands_device_id)
            case ("cuda", "cpu"):
                execution = ExecutionCUDA()
            case (None, "cpu") | ("cpu", _):
                execution = ExecutionCPU()
            case (ExecutionCUDA(), int()):
                # If operands are on a CUDA device, use the same device for execution.
                execution = dataclasses.replace(execution, device_id=operands_device_id)
            case (ExecutionCPU(), _) | (ExecutionCUDA(), "cpu"):
                pass
            case _:
                raise ValueError(
                    f"{self.__class__.__name__}.execution must be one of ExecutionCUDA, ExecutionCPU, None, 'cuda', or 'cpu'."
                )
        assert isinstance(execution, (ExecutionCPU, ExecutionCUDA))
        self.execution = execution

        self._operands_device_id = operands_device_id
        self._operands_package = utils.get_operands_package(operands)
        self._internal_op_package = self._internal_operand_package(self._operands_package)
        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)

        self._logger.info(
            f"The input tensors are located on device {operands_device_id}, and the execution space "
            f"is {self.execution.name}, with device {getattr(self.execution, 'device_id', 'cpu')}."
        )

        self._logger.info(
            f"The specified stream for the {self.__class__.__name__} constructor is "
            f"{(exec_stream_holder or operand_stream_holder) and getattr(exec_stream_holder or operand_stream_holder, 'obj', None)}."  # noqa: E501
        )

        operands_backup: list[utils.TensorHolder | None] = [None] * len(operands)
        for i in range(len(operands)):
            # Copy the operand to execution_space's device if needed.
            operands[i], operands_backup[i] = copy_operand_perhaps(
                None,
                operands[i],
                operand_stream_holder,
                getattr(self.execution, "device_id", "cpu"),
                self._operands_device_id,
            )
        self._operands = operands
        self._operands_backup = operands_backup

        # The result's package and device.
        self._result_class = self._operands[0].__class__

        # Set blocking or non-blocking behavior.
        self._blocking = self._options.blocking != "auto" or self._operands_device_id == "cpu" or self.execution.name == "cpu"
        if self._blocking:
            call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )
        self._call_prologue = call_prologue

        # Set memory allocator.
        allocator: memory.BaseCUDAMemoryManager | memory.BaseCUDAMemoryManagerAsync | None
        match self.execution:
            case ExecutionCUDA():
                allocator = (
                    memory._MEMORY_MANAGER[self._internal_op_package](self.execution.device_id, self._logger)
                    if self._options.allocator is None
                    else self._options.allocator
                )
            case ExecutionCPU() | _:
                allocator = None  # currently, the nvpl/fftw does not support custom workspace allocation
        self._allocator = allocator

        self._has_plan = False

    def _internal_operand_package(self, package_name: str) -> str:
        if self.execution.name == "cuda":
            return package_name if package_name != "numpy" else "cuda"
        else:
            return package_name if package_name != "cupy" else "cupy_host"

    def _get_or_create_stream_maybe(
        self, stream: utils.AnyStream
    ) -> tuple[utils.StreamHolder | None, utils.StreamHolder | None]:
        """Return a 2-tuple of Stream | None: one for execution space, one for input space.

        The first stream should be used for everything in the execution space: doing work,
        allocating workspace, allocating input/output buffers.

        The second stream should be used whenever data is being moved between the input and
        output spaces: copying data to/from the input/output tensors.

        NOTE: If two streams are returned, they will be the same stream.
        """
        if self.execution.name == "cuda":
            stream_holder = utils.get_or_create_stream(self.execution.device_id, stream, self._internal_op_package)
            return stream_holder, stream_holder
        elif isinstance(self._operands_device_id, int):
            operand_device_steam = utils.get_or_create_stream(self._operands_device_id, stream, self._operands_package)
            return None, operand_device_steam
        else:
            return None, None

    # input checks

    def _check_valid_operands(self, *args, **kwargs):
        """
        Check if the operands are available for the operation.
        """
        what = kwargs["what"]
        if self._operands is None:
            raise RuntimeError(
                f"{what} cannot be performed if the operands have been set to None. Use reset_operands() to set the "
                f"desired input before using performing the {what.lower()}."
            )

    def _check_planned(self, *args, **kwargs):
        what = kwargs["what"]
        if not self._has_plan:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    # execution

    @abc.abstractmethod
    def _execute(self):
        """Perform the main functionality of this :class:`StatefulAPI` instance without
        safety checks."""
        msg = f"{self.__name__}._execute() is not implemented."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def plan(self):
        """Plan the main functionality of this :class:`StatefulAPI` instance."""
        msg = f"{self.__class__.__name__}.plan() is not implemented."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def reset_operands(self):
        """Reset the operands held by this :class:`StatefulAPI` instance."""
        msg = f"{self.__class__.__name__}.reset_operands() is not implemented."
        raise NotImplementedError(msg)


class HasWorkspaceMemory(contextlib.AbstractContextManager):
    """A base class for APIs which need to allocate a working buffer in memory."""

    def _allocate_workspace_memory_perhaps(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been
        done.
        """
        raise NotImplementedError

    def _release_workspace_memory_perhaps(self, release_workspace: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def free(self):
        """Free the resources of this :class:`StatefulAPI` instance."""
        msg = f"{self.__name__}.free() is not implemented."
        raise NotImplementedError(msg)
