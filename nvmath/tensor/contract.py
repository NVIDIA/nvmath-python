# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
import numpy as np
from .. import memory
from ..bindings import cutensor
from ..internal import formatters
from ..internal import utils
from ..internal import tensor_wrapper
from .._utils import CudaDataType
from ._internal import einsum_parser
from ..internal.typemaps import NAME_TO_DATA_TYPE, DATA_TYPE_TO_NAME
from ._internal.typemaps import get_default_compute_type_from_dtype_name, get_supported_compute_types
from ._internal.cutensor_config_ifc import ContractionPlanPreference
from ._configuration import ContractionOptions, ExecutionCUDA

__all__ = [
    "BinaryContraction",
    "TernaryContraction",
    "ContractionPlanPreference",
    "ComputeDesc",
    "binary_contraction",
    "ternary_contraction",
    "Operator",
    "tensor_qualifiers_dtype",
]


Operator = cutensor.Operator

ComputeDesc = cutensor.ComputeDesc

tensor_qualifiers_dtype = np.int32

# As of cuTensor 2.3.1, only the following operators are supported in the contraction APIs
OPERATORS_SUPPORTED = {Operator.OP_IDENTITY, Operator.OP_CONJ}


def _compute_pointer_alignment(ptr: int) -> int:
    """
    Compute the pointer alignment for the given pointer.

    Args:
        ptr: Pointer address as integer

    Returns:
        The alignment value (256, 128, 64, 32, 16, 8, 4, 2, or 1)
    """
    return 256 if ptr == 0 else min(ptr & -ptr, 256)


SHARED_CONTRACTION_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_CONTRACTION_DOCUMENTATION.update(
    {
        "expr": """\
The einsum expression to perform the contraction.
""".replace("\n", " "),
        #
        "a": """\
A tensor representing the first operand to the tensor contraction. The currently supported types
are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.""".replace("\n", " "),
        #
        "b": """\
A tensor representing the second operand to the tensor contraction. The currently supported types
are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.""".replace("\n", " "),
        #
        "c": """\
A tensor representing the third operand to the tensor contraction. The currently supported types
are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.""".replace("\n", " "),
        #
        "addend": """\
(Optional) A tensor representing the operand to add to the tensor contraction result (fused operation in cuTensor).
The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`,
and :class:`torch.Tensor`.""".replace("\n", " "),
        #
        "alpha": """\
The scale factor for the tensor contraction term as a real or complex number. The default is
:math:`1.0`.""".replace("\n", " "),
        #
        "beta": """\
The scale factor for the tensor addition term as a real or complex number. A value for `beta` must be provided if
the operand to be added is specified.""".replace("\n", " "),
        #
        "qualifiers": """\
If desired, specify the operators as a :class:`numpy.ndarray` of dtype :class:`~nvmath.tensor.tensor_qualifiers_dtype`
with the same length as the number of operands in the contraction expression plus one (for the operand to be added).
All elements must be valid :class:`~nvmath.tensor.Operator` objects. See
:ref:`matrix-tensor-qualifiers` for the motivation behind qualifiers.""".replace("\n", " "),
        #
        "options": """\
Specify options for the tensor contraction as a :class:`~nvmath.tensor.ContractionOptions` object. Alternatively,
a `dict` containing the parameters for the ``ContractionOptions`` constructor can also be provided. If not specified, the
value will be set to the default-constructed ``ContractionOptions`` object.""".replace("\n", " "),
        #
        "execution": """\
Specify execution space options for the tensor contraction as a :class:`ExecutionCUDA` object or a string 'cuda'.
Alternatively, a `dict` containing 'name' key set to 'cuda' and the additional parameters for the ``ExecutionCUDA``
constructor can also be provided. If not provided, the execution space will be selected to match operand's storage if
the operands are on the GPU. If the operands are on the CPU and execution space is not provided, the execution space
will be a default-constructed :class:`ExecutionCUDA` object with device_id = 0.""".replace("\n", " "),
        #
        "out": """\
(Optional) The output tensor to store the result of the contraction. Must be a :class:`numpy.ndarray`, \
:class:`cupy.ndarray`, or :class:`torch.Tensor` object and must be on the same device as the input operands. \
If not specified, the result will be returned on the same device as the input operands.

        .. note::

            The support of output tensor in the API is experimental and subject to change in future versions
            without prior notice.

""".strip(),
        #
        "result": """\
The result of the specified contraction, which remains on the same device and belong to the
same package as the input operands. """.replace("\n", " "),
    }
)


class InvalidContractionState(Exception):
    pass


@utils.docstring_decorator(SHARED_CONTRACTION_DOCUMENTATION, skip_missing=False)
class _ElementaryContraction:
    """
    Pairwise contraction:
        O = A * B + C
    Ternary contraction:
        O = A * B * C + D
    """

    def __init__(self, expr, a, b, *, c=None, d=None, out=None, qualifiers=None, options=None, execution=None, stream=None):
        """Binary & Ternary Contraction"""

        version = cutensor.get_version()
        if version < 20301:
            raise RuntimeError(
                f"cuTensor version {version} is detected, which is lower than the minimum required "
                f"version 2.3.1 for nvmath.tensor module. please upgrade cuTensor to a compatible version."
            )

        self.expr = expr

        # Process options.
        self.options: Any = utils.check_or_create_options(ContractionOptions, options, "elementary contraction options")
        self.blocking = self.options.blocking
        self.logger = self.options.logger if self.options.logger is not None else logging.getLogger()

        # Process operands & einsum expression
        self.a, self.b = tensor_wrapper.wrap_operands([a, b])
        input_operand_class = self.a.__class__
        self.input_package = utils.get_operands_package([self.a, self.b])
        inputs, output = einsum_parser.parse_einsum_str(expr)
        self.num_inputs = len(inputs)

        if self.num_inputs == 2:
            assert d is None, f"Internal error: Binary contraction {expr} cannot have a fourth operand"
        elif self.num_inputs == 3:
            assert c is not None, f"Internal error: Ternary contraction {expr} must have a third operand"
        else:
            raise NotImplementedError("Only binary and ternary contractions are supported")

        wrapped_operands = [self.a, self.b]
        for op_name, op in zip(["c", "d"], [c, d], strict=False):
            if op is not None:
                op = tensor_wrapper.wrap_operand(op)
                if op.name != self.input_package:
                    raise ValueError(f"The operand {op_name} must be a {self.input_package} tensor")
                wrapped_operands.append(op)
            setattr(self, op_name, op)

        if self.input_package == "numpy":
            self.internal_package = "cuda"
        else:
            self.internal_package = self.input_package
        tensor_wrapper.maybe_register_package(self.internal_package)

        self.input_device_id = utils.get_operands_device_id(wrapped_operands)

        if execution is None:
            self.execution = ExecutionCUDA()
        else:
            self.execution = utils.check_or_create_one_of_options(
                (ExecutionCUDA,),
                execution,
                "execution options",
            )
        # TODO: cutensor supports R_64F (A) C_64F (B) C_64F (C) combination (and inverse)
        # https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreatecontraction
        self.data_type = utils.get_operands_dtype(wrapped_operands)
        self.cuda_data_type = NAME_TO_DATA_TYPE[self.data_type]

        # Parse compute descriptor
        if self.options.compute_type is None:
            self.compute_type = get_default_compute_type_from_dtype_name(self.data_type)
        elif isinstance(self.options.compute_type, int):
            # make sure compute type is valid
            if self.options.compute_type not in get_supported_compute_types(self.data_type):
                raise ValueError(f"Invalid compute type: {self.options.compute_type} for data type: {self.data_type}")
            self.compute_type = self.options.compute_type
        else:
            raise ValueError(f"Invalid compute type: {self.options.compute_type}")

        if self.input_device_id == "cpu":
            self.execution_device_id = self.execution.device_id
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
            self.a = self.a.to(self.execution_device_id, stream_holder)
            self.b = self.b.to(self.execution_device_id, stream_holder)
            if self.c is not None:
                self.c = self.c.to(self.execution_device_id, stream_holder)
            if self.d is not None:
                self.d = self.d.to(self.execution_device_id, stream_holder)
        else:
            self.execution_device_id = self.input_device_id
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)

        if qualifiers is None:
            self.qualifiers = np.full(self.num_inputs + 1, cutensor.Operator.OP_IDENTITY, dtype=np.int32)  # size of enum value
        else:
            self.qualifiers = np.asarray(qualifiers, dtype=np.int32)
            if self.qualifiers.size != self.num_inputs + 1:
                if self.num_inputs == 2:
                    message = f"The qualifiers must be a numpy array of length {self.num_inputs + 1}\
                              corresponding to the operands a, b and c"
                else:
                    message = f"The qualifiers must be a numpy array of length {self.num_inputs + 1}\
                              corresponding to the operands a, b, c and d"
                raise ValueError(message)
            if self.qualifiers[self.num_inputs] != cutensor.Operator.OP_IDENTITY:
                raise ValueError(
                    f"The operand for the offset must be the identity operator, found {self.qualifiers[self.num_inputs]}"
                )
            if self.num_inputs == 2:
                iterator = zip(["a", "b"], self.qualifiers[:-1], strict=False)
            else:
                iterator = zip(["a", "b", "c"], self.qualifiers[:-1], strict=False)
            for op_name, qualifier in iterator:
                if qualifier not in OPERATORS_SUPPORTED:
                    raise ValueError(
                        f"Each operator must be a valid cutensor operator, "
                        f"currently only support {OPERATORS_SUPPORTED}, "
                        f"got {qualifier}."
                    )
                if qualifier == cutensor.Operator.OP_CONJ:
                    operand = getattr(self, op_name)
                    if "complex" not in operand.dtype:
                        raise ValueError(f"The operand {op_name} must be a complex tensor to use the conjugate operator.")

        # Set memory allocator.
        self.allocator = (
            self.options.allocator
            if self.options.allocator is not None
            else memory._MEMORY_MANAGER[self.internal_package](self.execution_device_id, self.logger)
        )

        self.memory_limit = utils.get_memory_limit_from_device_id(self.options.memory_limit, self.execution_device_id)

        self.tensor_descs = {}

        self.input_modes, self.output_modes, _, size_dict = einsum_parser.parse_elementary_einsum(
            inputs, output, self.a, self.b, self.c
        )[:4]

        output_shape = [size_dict[mode] for mode in self.output_modes]

        # self.out is the output tensor that will be used for the execution
        # self.out_return is the output tensor that will be returned by the execute method
        self.output_provided = out is not None
        if self.output_provided:
            out = tensor_wrapper.wrap_operand(out)
            if out.name != self.input_package:
                raise ValueError(f"The output operand out must be a {self.input_package} tensor")
            if out.device_id != self.input_device_id:
                raise ValueError("The output operand out must be on the same device as the input operands.")
            self.out_return = out
            if out.device_id == self.execution_device_id:
                self.out = out
            else:
                self.out = out.to(self.execution_device_id, stream_holder)
        else:
            self.out = utils.create_empty_tensor(
                self.a.__class__, output_shape, self.data_type, self.execution_device_id, stream_holder, False
            )
            if self.input_device_id == self.execution_device_id:
                self.out_return = self.out
            else:
                tmp_stream_holder = None if self.input_device_id == "cpu" else stream_holder
                self.out_return = utils.create_empty_tensor(
                    input_operand_class, output_shape, self.data_type, self.input_device_id, tmp_stream_holder, False
                )

        with utils.device_ctx(self.execution_device_id):
            if self.options.handle is not None:
                self.own_handle = False
                self.handle = self.options.handle
                self.logger.info(f"The library handle has been set to the specified value: {self.handle}.")
            else:
                self.own_handle = True
                self.handle = cutensor.create()
                self.logger.info(f"The library handle has been created: {self.handle}.")

        self.valid_state = True

        # Parse tensor descriptors
        self.operands_info = {}
        self.pointer_alignment = {}
        for op_name in ["a", "b", "c", "d", "out"]:
            op = getattr(self, op_name)
            if op is not None:
                self.pointer_alignment[op_name] = _compute_pointer_alignment(op.data_ptr)
                self.tensor_descs[op_name] = cutensor.create_tensor_descriptor(
                    self.handle, len(op.shape), op.shape, op.strides, self.cuda_data_type, self.pointer_alignment[op_name]
                )
                self.operands_info[op_name] = {
                    "dtype": op.dtype,
                    "shape": op.shape,
                    "strides": op.strides,
                }

        # Create contraction descriptor
        if self.num_inputs == 2:
            self.contraction_desc = cutensor.create_contraction(
                self.handle,
                self.tensor_descs["a"],
                self.input_modes[0],
                self.qualifiers[0],
                self.tensor_descs["b"],
                self.input_modes[1],
                self.qualifiers[1],
                self.tensor_descs["out"]
                if c is None
                else self.tensor_descs["c"],  # if c is set to None, then C descriptor is the same as the out descriptor
                self.output_modes,  # NOTE: currently assuming c has the same output modes as the out
                self.qualifiers[2],  # only identity operator is supported for c
                self.tensor_descs["out"],
                self.output_modes,
                self.compute_type,
            )
        else:
            self.contraction_desc = cutensor.create_contraction_trinary(
                self.handle,
                self.tensor_descs["a"],
                self.input_modes[0],
                self.qualifiers[0],
                self.tensor_descs["b"],
                self.input_modes[1],
                self.qualifiers[1],
                self.tensor_descs["c"],
                self.input_modes[2],
                self.qualifiers[2],
                self.tensor_descs["out"]
                if d is None
                else self.tensor_descs["d"],  # if d is set to None, then D descriptor is the same as the out descriptor
                self.output_modes,
                self.qualifiers[3],  # only identity operator is supported for d
                self.tensor_descs["out"],
                self.output_modes,
                self.compute_type,
            )

        scalar_dtype = cutensor.get_operation_descriptor_attribute_dtype(cutensor.OperationDescriptorAttribute.SCALAR_TYPE)
        scalar_dtype_buffer = np.empty(1, dtype=scalar_dtype)
        cutensor.operation_descriptor_get_attribute(
            self.handle,
            self.contraction_desc,
            cutensor.OperationDescriptorAttribute.SCALAR_TYPE,
            scalar_dtype_buffer.ctypes.data,
            scalar_dtype_buffer.itemsize,
        )
        self.scalar_type = CudaDataType(scalar_dtype_buffer.item())

        self.alpha = np.empty(1, dtype=DATA_TYPE_TO_NAME[self.scalar_type])
        self.beta = np.empty(1, dtype=DATA_TYPE_TO_NAME[self.scalar_type])

        self.contraction_planned = False
        self.plan_preference_ptr = cutensor.create_plan_preference(self.handle, cutensor.Algo.DEFAULT, cutensor.JitMode.NONE)
        self._plan_preference = ContractionPlanPreference(self)
        self.plan_ptr = None

        self.workspace_ptr = None
        self.workspace_allocated_size = 0
        self.workspace_size = None
        self.workspace_stream = None
        self.workspace_allocated_here = False

    def _check_valid_contraction(self, *args, **kwargs):
        """
        Check if the ElementaryContraction object is alive and well.
        """
        if not self.valid_state:
            raise InvalidContractionState("The ElementaryContraction object cannot be used after resources are free'd")

    def _check_valid_operands(self, *args, **kwargs):
        """
        Check if the operands are available for the operation.
        """
        what = kwargs["what"]
        if self.num_inputs == 2:
            if self.a is None or self.b is None:
                raise RuntimeError(
                    f"{what} cannot be performed if a or b have been set to None "
                    f"for pairwise contraction. Use reset_operands() to set the "
                    f"desired input before using performing the {what.lower()}."
                )
        else:
            if self.a is None or self.b is None or self.c is None:
                raise RuntimeError(
                    f"{what} cannot be performed if a, b, or c have been set to None "
                    f"for ternary contraction. Use reset_operands() to set the "
                    f"desired input before using performing the {what.lower()}."
                )

        if self.output_provided and self.out is None:
            raise RuntimeError(
                f"{what} cannot be performed if out has been set to None. Use reset_operands() to set the "
                f"desired input before using performing the {what.lower()}."
            )

    def _free_plan_resources(self, exception: Exception | None = None) -> bool:
        """
        Free resources allocated in planning.
        """

        if self.plan_ptr is not None:
            cutensor.destroy_plan(self.plan_ptr)
            self.plan_ptr = None

        if self.plan_preference_ptr is not None:
            cutensor.destroy_plan_preference(self.plan_preference_ptr)
            self.plan_preference_ptr = None

        self.contraction_planned = False
        return True

    def _check_planned(self, *args, **kwargs):
        what = kwargs["what"]
        if not self.contraction_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _free_workspace_memory(self, exception: Exception | None = None) -> bool:
        """
        Free workspace by releasing the MemoryPointer object.
        """
        if self.workspace_ptr is None:
            return True

        self.workspace_ptr = None
        self.workspace_allocated_size = 0
        self.logger.debug("[_free_workspace_memory] The workspace has been released.")

        return True

    def _reset_workspace_allocation_tracking(self):
        """
        Reset workspace allocation tracking attributes to False at the end of the
        methods where workspace memory is potentially allocated. This is necessary
        to prevent any exceptions raised before method entry from using stale
        tracking values.
        """
        self.workspace_allocated_here = False

    @utils.precondition(_check_valid_contraction)
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
        self._release_workspace_memory_perhaps(release_workspace=self.workspace_allocated_here)
        self._reset_workspace_allocation_tracking()
        return True

    @utils.precondition(_check_valid_contraction)
    @utils.precondition(_check_planned, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator.
        """

        assert self.workspace_size is not None, "Internal Error."
        assert self.workspace_allocated_here is False, "Internal Error."

        if self.workspace_size == 0:  # For performance, bypass allocator for workspace size == 0.
            self.workspace_ptr = memory.MemoryPointer(0, 0, finalizer=None)
        else:
            self.logger.debug("Allocating workspace for performing the tensor contraction...")
            with utils.device_ctx(self.execution_device_id), stream_holder.ctx:
                try:
                    if isinstance(self.allocator, memory.BaseCUDAMemoryManagerAsync):
                        self.workspace_ptr = self.allocator.memalloc_async(self.workspace_size, stream_holder.obj)
                    else:
                        self.workspace_ptr = self.allocator.memalloc(self.workspace_size)
                    self.workspace_allocated_here = True
                except TypeError as e:
                    message = (
                        "The method 'memalloc' in the allocator object must conform to the interface in the "
                        "'BaseCUDAMemoryManager' protocol."
                    )
                    raise TypeError(message) from e

        self.workspace_allocated_size = self.workspace_size
        self.workspace_stream = stream_holder.obj
        self.logger.debug(
            f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size)} in the context "
            f"of stream {self.workspace_stream}."
        )

    def _allocate_workspace_memory_perhaps(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't
        already been done.
        """

        if self.workspace_ptr is not None and self.workspace_allocated_size >= self.workspace_size:
            return

        return self._allocate_workspace_memory(stream_holder)

    @property
    def plan_preference(self):
        """
        An accessor to configure or query the contraction planning phase
        attributes.

        Returns:
            A :class:`ContractionPlanPreference` object, whose attributes can be set (or
            queried) to configure the planning phase.

        .. seealso::
            :class:`ContractionPlanPreference`, :meth:`plan`.
        """
        return self._plan_preference

    @utils.precondition(_check_valid_contraction)
    def reset_operands(self, a=None, b=None, *, c=None, d=None, out=None, stream=None):
        if self.num_inputs == 2 and d is not None:
            raise RuntimeError("Internal Error: For pairwise contractions, d can not be set.")

        stream_holder = None  # lazy initialization
        for op_name, op in zip(["a", "b", "c", "d", "out"], [a, b, c, d, out], strict=False):
            if op is None:
                if op_name == "out":
                    if self.output_provided:
                        self.out_return = None
                        self.out = None
                    else:
                        # if out is not provided during initialization,
                        # we don't do anything with it
                        continue
                else:
                    setattr(self, op_name, None)
                self.logger.info(f"operand {op_name} has been reset to None.")
                continue
            tensor_info = self.operands_info.get(op_name)
            if tensor_info is None:
                raise ValueError(
                    f"operand {op_name} was not specified during the initialization "
                    f"of the ElementaryContraction object and therefore can not be reset "
                    f"to a concrete tensor."
                )
            op = tensor_wrapper.wrap_operand(op)
            if op.name != self.input_package:
                raise ValueError(f"The operand {op_name} must be a {self.input_package} tensor")
            if op.device_id != self.input_device_id:
                raise ValueError(
                    f"The operand {op_name} must be on the same device "
                    f"as the operands provided during the initialization of the "
                    f"ElementaryContraction object."
                )

            for attr, value in tensor_info.items():
                if getattr(op, attr) != value:
                    raise ValueError(
                        f"The operand {op_name} must have the same {attr} "
                        f"as the one specified during the initialization of the "
                        f"ElementaryContraction object."
                    )

            if op_name == "out":
                self.out_return = op

            if op.device_id != self.execution_device_id:
                if stream_holder is None:
                    stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
                if op_name == "out" and self.out is not None:
                    # if out_name is "out" and we own a valid self.out,
                    # we can directly reuse it here
                    op = self.out
                else:
                    op = op.to(self.execution_device_id, stream_holder)

            if _compute_pointer_alignment(op.data_ptr) != self.pointer_alignment[op_name]:
                raise ValueError(
                    f"The operand {op_name} must have the same pointer alignment "
                    f"as the one specified during the initialization of the "
                    f"ElementaryContraction object."
                )

            setattr(self, op_name, op)
            self.logger.info(f"operand {op_name} has been reset to the new operand provided.")
        return

    @utils.precondition(_check_valid_contraction)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(self, *, stream=None):
        """
        Plan the tensor contraction. The planning phase can be optionally configured through
        the property :attr:`plan_preference` (an object of type
        :class:`ContractionPlanPreference`).

        Args:
            stream: {stream}

        .. seealso::
            :attr:`plan_preference`, :class:`ContractionPlanPreference`.

        Note:
            If the :attr:`plan_preference` has been updated, a :meth:`plan` call is
            required to apply the changes.
        """
        log_info = self.logger.isEnabledFor(logging.INFO)

        # A new plan needs to be created at each plan() call
        if self.plan_ptr is not None:
            cutensor.destroy_plan(self.plan_ptr)
            self.plan_ptr = None

        if log_info:
            self.logger.info("= PLANNING PHASE =")
        stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
        required_workspace_size_buffer = np.empty(
            1, dtype=cutensor.get_plan_attribute_dtype(cutensor.PlanAttribute.REQUIRED_WORKSPACE)
        )

        with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            self.plan_ptr = cutensor.create_plan(
                self.handle, self.contraction_desc, self.plan_preference_ptr, self.memory_limit
            )
            cutensor.plan_get_attribute(
                self.handle,
                self.plan_ptr,
                cutensor.PlanAttribute.REQUIRED_WORKSPACE,
                required_workspace_size_buffer.ctypes.data,
                required_workspace_size_buffer.itemsize,
            )

        if log_info and elapsed.data is not None:
            self.logger.info(f"The planning phase took {elapsed.data:.3f} ms to complete.")

        self.workspace_size = required_workspace_size_buffer.item()
        self.contraction_planned = True

    @utils.precondition(_check_valid_contraction)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operands, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps_wrapper, method=True)
    def execute(self, *, alpha=1.0, beta=None, release_workspace=False, stream=None):
        """
        Execute a prepared tensor contraction.

        Args:
            alpha: {alpha}

            beta: {beta}

            release_workspace: {release_workspace}

            stream: {stream}

        Returns:
           {result}
        """
        if beta is None:
            if self.num_inputs == 2 and self.c is not None:
                raise ValueError("beta must be set when c is specified in a binary contraction")
            elif self.num_inputs == 3 and self.d is not None:
                raise ValueError("beta must be set when d is specified in a ternary contraction")
            beta = 0.0
        else:
            if self.num_inputs == 2 and self.c is None:
                raise ValueError("For binary contraction, beta can only be set if c is specified")
            elif self.num_inputs == 3 and self.d is None:
                raise ValueError("For ternary contraction, beta can only be set if d is specified")

        log_info = self.logger.isEnabledFor(logging.INFO)

        self.alpha[0] = alpha
        self.beta[0] = beta

        if log_info:
            self.logger.info("= EXECUTION PHASE =")
        stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
        if log_info:
            self.logger.info(f"The specified stream for execute() is {stream_holder.obj}.")

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(stream_holder)

        raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)

        with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            if self.num_inputs == 2:
                cutensor.contract(
                    self.handle,
                    self.plan_ptr,
                    self.alpha.ctypes.data,
                    self.a.data_ptr,
                    self.b.data_ptr,
                    self.beta.ctypes.data,
                    self.c.data_ptr if self.c is not None else self.out.data_ptr,
                    self.out.data_ptr,
                    raw_workspace_ptr,
                    self.workspace_size,
                    stream_holder.ptr,
                )
            else:
                cutensor.contract_trinary(
                    self.handle,
                    self.plan_ptr,
                    self.alpha.ctypes.data,
                    self.a.data_ptr,
                    self.b.data_ptr,
                    self.c.data_ptr,
                    self.beta.ctypes.data,
                    self.d.data_ptr if self.d is not None else self.out.data_ptr,
                    self.out.data_ptr,
                    raw_workspace_ptr,
                    self.workspace_size,
                    stream_holder.ptr,
                )

        if log_info and elapsed.data is not None:
            self.logger.info(f"The tensor contraction calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if requested.
        if release_workspace:
            self._release_workspace_memory_perhaps(True)

        self._reset_workspace_allocation_tracking()

        if self.out.device_id != self.out_return.device_id:
            self.out_return.copy_(self.out, stream_holder)
        return self.out_return.tensor

    @utils.precondition(_check_valid_contraction)
    def free(self):
        """Free tensor contraction resources.

        It is recommended that the contraction object be used
        within a context, but if it is not possible then this method must be
        called explicitly to ensure that the tensor contraction resources
        (especially internal library objects) are properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            if self.last_compute_event is not None and self.workspace_stream is not None:
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            self._free_plan_resources()

            class_name = self.__class__.__name__
            # Free handle if we own it.
            if self.handle is not None and self.own_handle:
                cutensor.destroy(self.handle)
                self.handle, self.own_handle = None, False

            if self.contraction_desc is not None:
                cutensor.destroy_operation_descriptor(self.contraction_desc)
                self.contraction_desc = None

            while self.tensor_descs:
                tensor_desc = self.tensor_descs.popitem()[1]
                cutensor.destroy_tensor_descriptor(tensor_desc)

        except Exception as e:
            self.logger.critical(f"Internal error: only part of the {class_name} object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info(f"The {class_name} object's resources have been released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()


@utils.docstring_decorator(SHARED_CONTRACTION_DOCUMENTATION, skip_missing=False)
class BinaryContraction(_ElementaryContraction):
    """
    Create a stateful object encapsulating the specified binary tensor contraction
    :math:`\\alpha a @ b + \\beta c` and the required resources to perform the operation.
    A stateful object can be used to amortize the cost of preparation (planning in the
    case of binary tensor contraction) across multiple executions (also see the
    :ref:`Stateful APIs<host api types>` section).

    The function-form API :func:`binary_contraction` is a convenient alternative to using
    stateful objects for *single* use (the user needs to perform just one tensor
    contraction, for example), in which case there is no possibility of amortizing
    preparatory costs. The function-form APIs are just convenience wrappers around
    the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific binary tensor contraction operation.
    3. **Execution**: Perform the tensor contraction computation with :meth:`execute`.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on what's happening in the various phases described above can be
    obtained by passing in a :class:`logging.Logger` object to :class:`ContractionOptions`
    or by setting the appropriate options in the root logger object,
    which is used by default:

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

        c: {addend}

        out: {out}

        qualifiers: {qualifiers}

        stream: {stream}

        options: {options}

        execution: {execution}

    .. seealso::
        :attr:`plan_preference`, :meth:`plan`, :meth:`reset_operands`, :meth:`execute`

    Examples:

        >>> import numpy as np
        >>> import nvmath

        Create two 3-D float64 ndarrays on the CPU:

        >>> M, N, K = 32, 32, 32
        >>> a = np.random.rand(M, N, K)
        >>> b = np.random.rand(N, K, M)

        We will define a binary tensor contraction operation.

        Create a BinaryContraction object encapsulating the problem specification above:

        >>> contraction = nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`ContractionOptions`).

        Next, plan the operation. Optionally, preferences can
        be specified for planning:

        >>> contraction.plan()

        Now execute the binary tensor contraction, and obtain the result `r1` as a NumPy
        ndarray.

        >>> r1 = contraction.execute()

        Finally, free the object's resources. To avoid having to explicitly making this
        call, it's recommended to use the BinaryContraction object as a context manager
        as shown below, if possible.

        >>> contraction.free()

        Note that all :class:`BinaryContraction` methods execute on the current
        stream by default. Alternatively, the `stream` argument can be used to run a
        method on a specified stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        Create a 3-D float64 CuPy ndarray on the GPU:

        >>> import cupy as cp
        >>> a = cp.random.rand(M, N, K)
        >>> b = cp.random.rand(N, K, M)

        Create an BinaryContraction object encapsulating the problem specification
        described earlier and use it as a context manager.

        >>> with nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b) as contraction:
        ...     contraction.plan()
        ...
        ...     # Execute the operation to get the first result.
        ...     r1 = contraction.execute()
        ...
        ...     # Update operands A and B in-place (see reset_operands() for an
        ...     # alternative).
        ...     a[:] = cp.random.rand(M, K)
        ...     b[:] = cp.random.rand(K, N)
        ...
        ...     # Execute the operation to get the new result.
        ...     r2 = contraction.execute()


        All the resources used by the object are released at the end of the block.

        Further examples can be found in the `nvmath/examples/tensor/contraction
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction>`_
        directory.
    """

    def __init__(self, expr, a, b, *, c=None, out=None, qualifiers=None, stream=None, options=None, execution=None):
        super().__init__(expr, a, b, c=c, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution)

    def reset_operands(self, a=None, b=None, *, c=None, out=None, stream=None):
        """
        Reset the operands held by this :class:`BinaryContraction` instance.

        This method has two use cases:
            (1) it can be used to provide new operands for execution when the original
                operands are on the CPU
            (2) it can be used to release the internal reference to the previous operands
                and make their memory available for other use by passing ``None`` for *all*
                arguments. In this case, this method must be called again to provide the
                desired operands before another call to execution APIs like :meth:`execute`.

        This method is not needed when the operands reside on the GPU and in-place
        operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

            - The shapes, strides, datatypes match those of the old ones.
            - The packages that the operands belong to match those of the old ones.
            - If input tensors are on GPU, the device must match.

        Args:
            a: {a}

            b: {b}

            c: {addend}

            out: {out}

            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import nvmath

            Create two 3-D float64 ndarrays on the GPU:

            >>> M, N, K = 128, 128, 256
            >>> a = cp.random.rand(M, K)
            >>> b = cp.random.rand(K, N)

            Create an binary contraction object as a context manager

            >>> with nvmath.tensor.BinaryContraction("ij,jk->ik", a, b) as contraction:
            ...     # Plan the operation.
            ...     algorithms = contraction.plan()
            ...
            ...     # Execute the contraction to get the first result.
            ...     r1 = contraction.execute()
            ...
            ...     # Reset the operands to new CuPy ndarrays.
            ...     a1 = cp.random.rand(M, K)
            ...     b1 = cp.random.rand(K, N)
            ...     contraction.reset_operands(a=a1, b=b1)
            ...
            ...     # Execute to get the new result corresponding to the updated operands.
            ...     r2 = contraction.execute()

            Note that if only a subset of operands are reset, the operands that are not
            reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is
            equivalent to updating the operands in-place, i.e, replacing
            ``contraction.reset_operand(a=a1, b=b1)`` with ``a[:]=a1`` and ``b[:]=b1``.
            Note that updating the operand in-place should be adopted with caution as it can
            only yield the expected result under the additional constraint below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction/example06_stateful_inplace.py>`_.
        """
        super().reset_operands(a=a, b=b, c=c, out=out, stream=stream)


@utils.docstring_decorator(SHARED_CONTRACTION_DOCUMENTATION, skip_missing=False)
class TernaryContraction(_ElementaryContraction):
    """
    Create a stateful object encapsulating the specified ternary tensor contraction
    :math:`\\alpha a @ b + \\beta c` and the required resources to perform the operation.
    A stateful object can be used to amortize the cost of preparation (planning in the
    case of ternary tensor contraction) across multiple executions (also see the
    :ref:`Stateful APIs<host api types>` section).

    The function-form API :func:`ternary_contraction` is a convenient alternative to using
    stateful objects for *single* use (the user needs to perform just one tensor
    contraction, for example), in which case there is no possibility of amortizing
    preparatory costs. The function-form APIs are just convenience wrappers around
    the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific ternary tensor contraction operation.
    3. **Execution**: Perform the tensor contraction computation with :meth:`execute`.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on what's happening in the various phases described above can be
    obtained by passing in a :class:`logging.Logger` object to :class:`ContractionOptions`
    or by setting the appropriate options in the root logger object,
    which is used by default:

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

        d: {addend}

        out: {out}

        qualifiers: {qualifiers}

        stream: {stream}

        options: {options}

        execution: {execution}

    .. seealso::
        :attr:`plan_preference`, :meth:`plan`, :meth:`reset_operands`, :meth:`execute`

    Examples:

        >>> import numpy as np
        >>> import nvmath

        Create three 3-D float64 ndarrays on the CPU:

        >>> M, N, K = 32, 32, 32
        >>> a = np.random.rand(M, N, K)
        >>> b = np.random.rand(N, K, M)
        >>> c = np.random.rand(M, N)

        We will define a ternary tensor contraction operation.

        Create a TernaryContraction object encapsulating the problem specification above:

        >>> expr = "ijk,jkl,ln->in"
        >>> contraction = nvmath.tensor.TernaryContraction(expr, a, b, c)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`ContractionOptions`).

        Next, plan the operation. Optionally, preferences can
        be specified for planning:

        >>> contraction.plan()

        Now execute the ternary tensor contraction, and obtain the result `r1` as
        a NumPy ndarray.

        >>> r1 = contraction.execute()

        Finally, free the object's resources. To avoid having to explicitly making this
        call, it's recommended to use the TernaryContraction object as a context manager
        as shown below, if possible.

        >>> contraction.free()

        Note that all :class:`TernaryContraction` methods execute on the current
        stream by default. Alternatively, the `stream` argument can be used to run a
        method on a specified stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        Create a 3-D float64 CuPy ndarray on the GPU:

        >>> import cupy as cp
        >>> a = cp.random.rand(M, N, K)
        >>> b = cp.random.rand(N, K, M)
        >>> c = cp.random.rand(M, N)

        Create an TernaryContraction object encapsulating the problem specification
        described earlier and use it as a context manager.

        >>> expr = "ijk,jkl,ln->in"
        >>> with nvmath.tensor.TernaryContraction(expr, a, b, c) as contraction:
        ...     contraction.plan()
        ...
        ...     # Execute the operation to get the first result.
        ...     r1 = contraction.execute()
        ...
        ...     # Update operands A, B and C in-place (see reset_operands() for an
        ...     # alternative).
        ...     a[:] = cp.random.rand(M, N, K)
        ...     b[:] = cp.random.rand(N, K, M)
        ...     c[:] = cp.random.rand(M, N)
        ...
        ...     # Execute the operation to get the new result.
        ...     r2 = contraction.execute()


        All the resources used by the object are released at the end of the block.

        Further examples can be found in the `nvmath/examples/tensor/contraction
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction>`_
        directory.
    """

    def __init__(self, expr, a, b, c, *, d=None, out=None, qualifiers=None, stream=None, options=None, execution=None):
        super().__init__(
            expr, a, b, c=c, d=d, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution
        )

    def reset_operands(self, a=None, b=None, c=None, *, d=None, out=None, stream=None):
        """
        Reset the operands held by this :class:`TernaryContraction` instance.

        This method has two use cases:
            (1) it can be used to provide new operands for execution when the original
                operands are on the CPU
            (2) it can be used to release the internal reference to the previous operands
                and make their memory available for other use by passing ``None`` for *all*
                arguments. In this case, this method must be called again to provide the
                desired operands before another call to execution APIs like :meth:`execute`.

        This method is not needed when the operands reside on the GPU and in-place
        operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

            - The shapes, strides, datatypes match those of the old ones.
            - The packages that the operands belong to match those of the old ones.
            - If input tensors are on GPU, the device must match.

        Args:
            a: {a}

            b: {b}

            c: {c}

            d: {addend}

            out: {out}

            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import nvmath

            Create two 3-D float64 ndarrays on the GPU:

            >>> M, N, K = 12, 16, 32
            >>> a = cp.random.rand(M, M, N)
            >>> b = cp.random.rand(N, K)
            >>> c = cp.random.rand(K, K)

            Create an ternary contraction object as a context manager

            >>> expr = "ijk,kl,lm->ijm"
            >>> with nvmath.tensor.TernaryContraction(expr, a, b, c) as contraction:
            ...     # Plan the operation.
            ...     algorithms = contraction.plan()
            ...
            ...     # Execute the contraction to get the first result.
            ...     r1 = contraction.execute()
            ...
            ...     # Reset the operands to new CuPy ndarrays.
            ...     a1 = cp.random.rand(M, M, N)
            ...     b1 = cp.random.rand(N, K)
            ...     c1 = cp.random.rand(K, K)
            ...     contraction.reset_operands(a=a1, b=b1, c=c1)
            ...
            ...     # Execute to get the new result corresponding to the updated operands.
            ...     r2 = contraction.execute()

            Note that if only a subset of operands are reset, the operands that are not
            reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands`
            is equivalent to updating the operands in-place, i.e, replacing
            ``contraction.reset_operand(a=a1, b=b1, c=c1)`` with ``a[:]=a1``
            and ``b[:]=b1`` and ``c[:]=c1``. Note that updating the operand in-place
            should be adopted with caution as it can only yield the expected result
            under the additional constraint below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction/example06_stateful_inplace.py>`_.
        """
        super().reset_operands(a=a, b=b, c=c, d=d, out=out, stream=stream)


@utils.docstring_decorator(SHARED_CONTRACTION_DOCUMENTATION, skip_missing=False)
def binary_contraction(
    expr, a, b, *, c=None, alpha=1.0, beta=None, out=None, qualifiers=None, stream=None, options=None, execution=None
):
    """
    Evaluate the Einstein summation convention for binary contraction on the operands.

    Explicit as well as implicit form is supported for the Einstein summation expression.

    Additionally, the binary contraction can be performed with
    an additional operand, which is added to the result with a scale factor.

    This function-form is a wrapper around the stateful
    :class:`BinaryContraction` object APIs and is meant for *single* use (the user needs
    to perform just one binary contraction, for example), in which case there is
    no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`ContractionOptions` or by setting the
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
        expr: {expr}

        a: {a}

        b: {b}

        c: {addend}

        alpha: {alpha}

        beta: {beta}

        out: {out}

        qualifiers: {qualifiers}

        stream: {stream}

        options: {options}

        execution: {execution}

    Returns:
        {result}

    .. seealso::
        :class:`BinaryContraction`, :func:`ternary_contraction`,
        :class:`TernaryContraction`, :class:`ContractionOptions`,
        :class:`ContractionPlanPreferences`

        For tensor network contraction with arbitrary number of operands including
        contraction path finding, see cuQuantum:

        - :external+cuquantum:py:func:`cuquantum.tensornet.contract`
        - :external+cuquantum:py:class:`cuquantum.tensornet.Network`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create three float32 ndarrays on the GPU:

        >>> M, N = 32, 64
        >>> a = cp.random.rand(M, M, N, N, dtype=cp.float32)
        >>> b = cp.random.rand(N, N, N, N, dtype=cp.float32)
        >>> c = cp.random.rand(M, M, N, N, dtype=cp.float32)

        Perform the operation :math:`\\alpha \\sum A[i,j,a,b] * B[a,b,c,d] +
        \\beta C[i,j,c,d]` using :func:`binary_contraction`.
        The result `r` is also a CuPy float32 ndarray:

        >>> r = nvmath.tensor.binary_contraction(
        ...     "ijab,abcd->ijcd", a, b, c=c, alpha=1.23, beta=0.74
        ... )

        The result is equivalent to:

        >>> r = 1.23 * cp.einsum("ijab,abcd->ijcd", a, b) + 0.74 * c

        Options can be provided to customize the operation:

        >>> compute_type = nvmath.bindings.cutensor.ComputeDesc.COMPUTE_3XTF32()
        >>> o = nvmath.tensor.ContractionOptions(compute_type=compute_type)
        >>> r = nvmath.tensor.binary_contraction("ijab,abcd->ijcd", a, b, options=o)

        See `ContractionOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly
        provided to the binary contraction operation. This can be done if the operands
        are computed on a different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...     a = cp.random.rand(M, M, N, N)
        ...     b = cp.random.rand(N, N, N, N)
        >>> r = nvmath.tensor.binary_contraction("ijab,abcd->ijcd", a, b, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create NumPy ndarrays on the CPU.

        >>> import numpy as np
        >>> a = np.random.rand(M, M, N, N)
        >>> b = np.random.rand(N, N, N, N)

        Provide the NumPy ndarrays to :func:`binary_contraction`, with the result
        also being a NumPy ndarray:

        >>> r = nvmath.tensor.binary_contraction("ijab,abcd->ijcd", a, b)

    Notes:
        - This function is a convenience wrapper around :class:`BinaryContraction` and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/tensor/contraction
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction>`_
    directory.
    """
    if c is None and beta is not None:
        raise ValueError("beta can only be set if c is specified in a binary contraction")
    elif c is not None and beta is None:
        raise ValueError("beta must be set when c is specified in a binary contraction")
    with BinaryContraction(
        expr, a, b, c=c, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution
    ) as contraction:
        contraction.plan()
        out = contraction.execute(alpha=alpha, beta=beta, stream=stream)
    return out


@utils.docstring_decorator(SHARED_CONTRACTION_DOCUMENTATION, skip_missing=False)
def ternary_contraction(
    expr, a, b, c, *, d=None, alpha=1.0, beta=None, out=None, qualifiers=None, stream=None, options=None, execution=None
):
    """
    Evaluate the Einstein summation convention for ternary contraction on the operands.

    Explicit as well as implicit form is supported for the Einstein summation expression.

    Additionally, the ternary contraction can be performed with
    an additional operand, which is added to the result with a scale factor.

    This function-form is a wrapper around the stateful
    :class:`TernaryContraction` object APIs and is meant for *single* use (the user needs
    to perform just one ternary contraction, for example), in which case there is
    no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`ContractionOptions` or by setting the
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
        expr: {expr}

        a: {a}

        b: {b}

        c: {c}

        d: {addend}

        alpha: {alpha}

        beta: {beta}

        out: {out}

        qualifiers: {qualifiers}

        stream: {stream}

        options: {options}

        execution: {execution}

    Returns:
        {result}

    .. seealso::
        :class:`TernaryContraction`, :func:`binary_contraction`,
        :class:`BinaryContraction`, :class:`ContractionOptions`,
        :class:`ContractionPlanPreferences`

        For tensor network contraction with arbitrary number of operands including
        contraction path finding, see cuQuantum:

        - :external+cuquantum:py:func:`cuquantum.tensornet.contract`
        - :external+cuquantum:py:class:`cuquantum.tensornet.Network`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create three float32 ndarrays on the GPU:

        >>> M, N, K = 16, 24, 32
        >>> a = cp.random.rand(M, M, dtype=cp.float32)
        >>> b = cp.random.rand(M, N, K, dtype=cp.float32)
        >>> c = cp.random.rand(N, K, M, dtype=cp.float32)
        >>> d = cp.random.rand(M, M, dtype=cp.float32)

        Perform the operation :math:`\\alpha \\sum A[i,j] * B[j,k,l] * C[k,l,m] +
        \\beta D[i,m]` using :func:`ternary_contraction`.
        The result `r` is also a CuPy float32 ndarray:

        >>> r = nvmath.tensor.ternary_contraction(
        ...     "ij,jkl,klm->im", a, b, c, d=d, alpha=0.63, beta=0.22
        ... )

        The result is equivalent to:

        >>> r = 0.63 * cp.einsum("ij,jkl,klm->im", a, b, c) + 0.22 * d

        Options can be provided to customize the operation:

        >>> compute_type = nvmath.bindings.cutensor.ComputeDesc.COMPUTE_3XTF32()
        >>> o = nvmath.tensor.ContractionOptions(compute_type=compute_type)
        >>> r = nvmath.tensor.ternary_contraction("ij,jkl,klm->im", a, b, c, options=o)

        See `ContractionOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly
        provided to the ternary contraction operation. This can be done if the operands
        are computed on a different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...     a = cp.random.rand(M, M, dtype=cp.float32)
        ...     b = cp.random.rand(M, N, K, dtype=cp.float32)
        ...     c = cp.random.rand(N, K, M, dtype=cp.float32)
        >>> r = nvmath.tensor.ternary_contraction("ij,jkl,klm->im", a, b, c, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create NumPy ndarrays on the CPU.

        >>> import numpy as np
        >>> a = np.random.rand(M, M)
        >>> b = np.random.rand(M, N, K)
        >>> c = np.random.rand(N, K, M)

        Provide the NumPy ndarrays to :func:`ternary_contraction`, with the result
        also being a NumPy ndarray:

        >>> r = nvmath.tensor.ternary_contraction("ij,jkl,klm->im", a, b, c)

    Notes:
        - This function is a convenience wrapper around :class:`TernaryContraction` and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/tensor/contraction
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction>`_
    directory.
    """
    if d is None and beta is not None:
        raise ValueError("beta can only be set if d is specified in a ternary contraction")
    elif d is not None and beta is None:
        raise ValueError("beta must be set when d is specified in a ternary contraction")
    with TernaryContraction(
        expr, a, b, c, d=d, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution
    ) as contraction:
        contraction.plan()
        out = contraction.execute(alpha=alpha, beta=beta, stream=stream)
    return out
