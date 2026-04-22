# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import numpy as np

from .. import memory
from .._utils import CudaDataType
from ..bindings import cutensor
from ..internal import formatters, tensor_wrapper, utils
from ..internal.typemaps import DATA_TYPE_TO_NAME, NAME_TO_DATA_TYPE
from ._configuration import ContractionOptions, ExecutionCUDA
from ._internal import cutensor_utils, einsum_parser
from ._internal.cutensor_config_ifc import ContractionPlanPreference
from ._internal.typemaps import get_compute_type_name, get_default_compute_type_from_dtype_name, get_supported_compute_types

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

# As of cuTensor 2.5.0, only the following operators are supported in the contraction APIs
OPERATORS_SUPPORTED = {Operator.OP_IDENTITY, Operator.OP_CONJ}

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
        #
        "reset_operands_semantics": """\
Semantics:
            - This method validates each new operand against its corresponding one
              set during the object's initialization.
              An operand is compatible if all of the following requirements are met:

              - The shapes, strides, and datatypes match those of the old one.
              - The package (e.g., cupy, torch, numpy) matches that of the old one.
              - The memory space (CPU or GPU) matches that of the old one.
              - The device matches that of the old one, if the operand is on GPU.
              - The pointer alignment must be the same or a multiple of the old one.

            - If the execution space matches the memory space of the operand:
              operand's reference is updated with no data copying.

            - If the execution space does not match the memory space of the operand:
              data is copied between different memory spaces.
""",
        #
        "reset_operands_unchecked": utils._reset_operand_unchecked_docstring(
            True, version_added="0.9.0", validation_examples="package match, data type match, pointer alignment match"
        ),
    }
)


class InvalidContractionState(Exception):
    pass


def _validate_contraction_preconditions(expr, a, b, c, d, qualifiers, options):
    """
    Validate preconditions for _ElementaryContraction initialization.

    This function checks all preconditions that can be validated before
    wrapping operands or allocating resources. It raises exceptions for invalid inputs.

    Args:
        expr: Einstein expression string
        a: First input operand
        b: Second input operand
        c: Optional third operand (for binary) or offset (for ternary)
        d: Optional offset operand (for ternary only)
        qualifiers: Optional qualifiers array
        options: Optional contraction options

    Returns:
        tuple: (num_inputs, inputs, output) from parsed expression
    """
    # Parse expression to determine number of inputs (validates expression format)
    inputs, output = einsum_parser.parse_einsum_str(expr)
    num_inputs = len(inputs)

    # Validate number of inputs, c/d operand consistency with expression type
    if num_inputs == 2:
        if d is not None:
            raise ValueError(f"Binary contraction '{expr}' (2 inputs) cannot have a 'd' operand. ")
    elif num_inputs == 3:
        if c is None:
            raise ValueError(f"Ternary contraction '{expr}' (3 inputs) requires 'c' operand (third multiplicand). ")
    else:
        raise NotImplementedError(
            f"Expression '{expr}' has {num_inputs} inputs. Only binary and ternary contractions are supported."
        )

    # Validate qualifiers structure (if provided)
    if qualifiers is not None:
        try:
            qualifiers_array = np.asarray(qualifiers, dtype=np.int32)
        except Exception as e:
            raise TypeError(f"Qualifiers must be array-like and convertible to int32: {e}") from e

        # Check array length
        expected_len = num_inputs + 1  # one per input + one for offset
        if qualifiers_array.size != expected_len:
            contraction_type = "binary" if num_inputs == 2 else "ternary"
            operand_names = "a, b, offset" if num_inputs == 2 else "a, b, c, offset"
            raise ValueError(
                f"The qualifiers must be a numpy array of length {expected_len} "
                f"(one per operand: {operand_names}), got {qualifiers_array.size}. "
                f"Expression '{expr}' is a {contraction_type} contraction."
            )

        # Validate offset qualifier must be identity
        if qualifiers_array[num_inputs] != cutensor.Operator.OP_IDENTITY:
            raise ValueError(f"The operand for the offset must be the identity operator, found {qualifiers_array[num_inputs]}")

        # Validate all input qualifiers are supported operators
        operand_names = ["a", "b", "c"][:num_inputs]
        for op_name, qualifier in zip(operand_names, qualifiers_array[:-1], strict=False):
            if qualifier not in OPERATORS_SUPPORTED:
                raise ValueError(
                    f"Each operator must be a valid cuTensor operator, "
                    f"currently only support {OPERATORS_SUPPORTED}, "
                    f"got {qualifier} for operand '{op_name}'."
                )

        # Validate qualifiers against operand dtypes (e.g., OP_CONJ requires complex)
        operands = [a, b, c][:num_inputs]
        for op_name, operand, qualifier in zip(operand_names, operands, qualifiers_array[:-1], strict=False):
            if qualifier == cutensor.Operator.OP_CONJ:
                # Check if operand has dtype attribute
                if not hasattr(operand, "dtype"):
                    raise TypeError(
                        f"Operand '{op_name}' must be array-like with a dtype attribute "
                        f"(numpy.ndarray, cupy.ndarray, or torch.Tensor)"
                    )

                # Check for complex dtype
                dtype_str = str(operand.dtype)
                if "complex" not in dtype_str:
                    raise ValueError(
                        f"Cannot apply OP_CONJ (conjugate) operator to operand '{op_name}' "
                        f"with dtype '{operand.dtype}'. Conjugate operator requires complex dtype."
                    )

    # Validate options structure (if provided)
    if options is not None:
        # Extract compute_type from options (could be dict or ContractionOptions object)
        if isinstance(options, dict):
            compute_type = options.get("compute_type")
        else:
            compute_type = getattr(options, "compute_type", None)

        # Validate compute_type is None or int
        if compute_type is not None and not isinstance(compute_type, int):
            raise ValueError(f"Invalid compute type: {compute_type}. compute_type must be None or an integer.")

    return num_inputs, inputs, output


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

        # Initialize valid_state to True here because this flag is currenttly only
        # used to check if the object has been free'd or not.
        # As far as freeing the object is concerned, upon initialization,
        # the object is already valid.
        self.valid_state = True

        # Track whether the user has called release_operands().
        self._operands_released = False

        # ========================================================================
        # Validate preconditions right away, if it fails, no state changes will be made
        # ========================================================================
        self.num_inputs, inputs, output = _validate_contraction_preconditions(expr, a, b, c, d, qualifiers, options)
        self.expr = expr

        # ========================================================================
        # Process options (needed for logging and configuration)
        # ========================================================================
        self.options: Any = utils.check_or_create_options(ContractionOptions, options, "elementary contraction options")
        self.logger = self.options.logger if self.options.logger is not None else logging.getLogger()
        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        # ========================================================================
        # Wrap and validate operands (a, b, and c, d if needed)
        # ========================================================================
        self.a, self.b = tensor_wrapper.wrap_operands([a, b])
        self.input_operand_class = self.a.__class__
        self.input_package = utils.get_operands_package([self.a, self.b])

        # Determine internal package (numpy -> cuda conversion)
        if self.input_package == "numpy":
            self.internal_package = "cuda"
        else:
            self.internal_package = self.input_package
        tensor_wrapper.maybe_register_package(self.internal_package)

        # Wrap optional operands c, d and validate package consistency
        wrapped_operands = [self.a, self.b]
        self.c_provided = c is not None
        self.d_provided = d is not None
        for op_name, op in zip(["c", "d"], [c, d], strict=False):
            if op is not None:
                op = tensor_wrapper.wrap_operand(op)
                if op.name != self.input_package:
                    raise ValueError(
                        f"operand has package '{op.name}' but expected '{self.input_package}'. "
                        f"All operands must be from the same tensor package."
                    )
                wrapped_operands.append(op)
            setattr(self, op_name, op)

        # ========================================================================
        # Setup qualifiers
        # ========================================================================
        # If here, preconditions were validated, we can safely create the qualifiers
        # or use the ones provided.
        if qualifiers is None:
            self.qualifiers = np.full(self.num_inputs + 1, cutensor.Operator.OP_IDENTITY, dtype=np.int32)  # size of enum value
        else:
            # validation of qualifiers done during preconditions' check
            self.qualifiers = np.asarray(qualifiers, dtype=np.int32)

        # ========================================================================
        # Setup execution environment and device management
        # ========================================================================
        self.input_device_id = utils.get_operands_device_id(wrapped_operands)
        self.blocking = self.options.blocking is True or self.input_device_id == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        if execution is None:
            self.execution = ExecutionCUDA()
        else:
            self.execution = utils.check_or_create_one_of_options(
                (ExecutionCUDA,),
                execution,
                "execution options",
            )

        if log_info:
            self.logger.info(
                f"The input tensor's memory space is {self.input_device_id}, and the execution space "
                f"is device {self.execution.device_id}."
            )

        # Determine execution device and create stream
        if self.input_device_id == "cpu":
            self.execution_device_id = self.execution.device_id
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
            # Transfer CPU operands to execution device
            self.a = self.a.to(self.execution_device_id, stream_holder)
            self.b = self.b.to(self.execution_device_id, stream_holder)
            if self.c is not None:
                self.c = self.c.to(self.execution_device_id, stream_holder)
            if self.d is not None:
                self.d = self.d.to(self.execution_device_id, stream_holder)
            if log_debug:
                self.logger.debug(f"The input tensors have been copied to the execution device: {self.execution_device_id}.")
        else:
            self.execution_device_id = self.input_device_id
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)

        # ========================================================================
        # Determine data types and compute configuration
        # ========================================================================
        # TODO: cutensor supports R_64F (A) C_64F (B) C_64F (C) combination (and inverse)
        # https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreatecontraction
        self.data_type = utils.get_operands_dtype(wrapped_operands)
        self.cuda_data_type = NAME_TO_DATA_TYPE[self.data_type]

        # Parse compute descriptor
        if self.options.compute_type is None:
            self.compute_type = get_default_compute_type_from_dtype_name(self.data_type)
        else:
            # Type validation done in preconditions, here check compatibility with data type
            if self.options.compute_type not in get_supported_compute_types(self.data_type):
                raise ValueError(f"Invalid compute type: {self.options.compute_type} for data type: {self.data_type}")
            self.compute_type = self.options.compute_type
            if self.compute_type == ComputeDesc.COMPUTE_8XINT8():
                # TODO: remove the check once cutensor requirement is bumped to 2.6.0
                version = cutensor.get_version()
                if version == 20500:
                    raise RuntimeError(
                        "The 8XINT8 compute type is not supported in cuTensor 2.5.0 due to a known bug. "
                        "Please upgrade cuTensor to a later version."
                    )

        if log_info:
            self.logger.info(f"The compute type is: {get_compute_type_name(self.compute_type)}.")
            qualifiers_message = (
                f"The contraction qualifiers are: "
                f"A = {cutensor.Operator(self.qualifiers[0]).name}, "
                f"B = {cutensor.Operator(self.qualifiers[1]).name}, "
                f"C = {cutensor.Operator(self.qualifiers[2]).name}"
            )
            if self.num_inputs == 3:
                qualifiers_message += f", D = {cutensor.Operator(self.qualifiers[3]).name}"
            self.logger.info(qualifiers_message)
        # ========================================================================
        # Parse einsum modes and setup output tensor
        # ========================================================================
        self.input_modes, self.output_modes, _, size_dict = einsum_parser.parse_elementary_einsum(
            inputs, output, self.a, self.b, self.c
        )[:4]

        self.output_shape = [size_dict[mode] for mode in self.output_modes]

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
                if log_debug:
                    self.logger.debug(f"The output tensor is copied to the execution device: {self.execution_device_id}.")
        else:
            self.out = self.out_return = None

        # ========================================================================
        # Setup memory management
        # ========================================================================
        self.allocator = (
            self.options.allocator
            if self.options.allocator is not None
            else memory._MEMORY_MANAGER[self.internal_package](self.execution_device_id, self.logger)
        )
        self.memory_limit = utils.get_memory_limit_from_device_id(self.options.memory_limit, self.execution_device_id)
        self.tensor_descs = {}

        # ========================================================================
        # Create cuTensor handle and descriptors
        # ========================================================================
        with utils.device_ctx(self.execution_device_id):
            if self.options.handle is not None:
                self.own_handle = False
                self.handle = self.options.handle
                self.logger.info(f"The library handle has been set to the specified value: {self.handle}.")
            else:
                self.own_handle = True
                self.handle = cutensor.create()
                self.logger.info(f"The library handle has been created: {self.handle}.")

        if self.num_inputs == 2:
            addend_name = "c"
            operands_names = ("a", "b", "out") if c is None else ("a", "b", "c", "out")
        else:
            addend_name = "d"
            operands_names = ("a", "b", "c", "out") if d is None else ("a", "b", "c", "d", "out")

        # Create tensor descriptors for all relevant operands
        for op_name in operands_names:
            op = getattr(self, op_name)
            if op is None:
                assert op_name == "out", "Internal Error: out should be None if not provided."
                self.tensor_descs[op_name] = cutensor_utils.TensorDescriptor.from_shape_and_dtype(
                    self.handle, self.output_shape, self.data_type
                )
            else:
                self.tensor_descs[op_name] = cutensor_utils.TensorDescriptor.from_tensor_holder(self.handle, op)
            if log_debug:
                self.logger.debug(
                    f"The tensor descriptor for operand {op_name} with shape {self.tensor_descs[op_name].shape}, "
                    f"strides {self.tensor_descs[op_name].strides}, dtype {self.tensor_descs[op_name].dtype}, "
                    f"and pointer alignment {self.tensor_descs[op_name].alignment} has been created."
                )

        if addend_name not in operands_names:
            # If addend is not specified, we can reuse the output descriptor for the addend
            self.tensor_descs[addend_name] = self.tensor_descs["out"]

        # ========================================================================
        # Create contraction descriptor
        # ========================================================================
        if self.num_inputs == 2:
            self.contraction_desc = cutensor.create_contraction(
                self.handle,
                self.tensor_descs["a"].ptr,
                self.input_modes[0],
                self.qualifiers[0],
                self.tensor_descs["b"].ptr,
                self.input_modes[1],
                self.qualifiers[1],
                self.tensor_descs["c"].ptr,
                self.output_modes,  # NOTE: currently assuming c has the same output modes as the out
                self.qualifiers[2],  # only identity operator is supported for c
                self.tensor_descs["out"].ptr,
                self.output_modes,
                self.compute_type,
            )
        else:
            self.contraction_desc = cutensor.create_contraction_trinary(
                self.handle,
                self.tensor_descs["a"].ptr,
                self.input_modes[0],
                self.qualifiers[0],
                self.tensor_descs["b"].ptr,
                self.input_modes[1],
                self.qualifiers[1],
                self.tensor_descs["c"].ptr,
                self.input_modes[2],
                self.qualifiers[2],
                self.tensor_descs["d"].ptr,
                self.output_modes,
                self.qualifiers[3],  # only identity operator is supported for d
                self.tensor_descs["out"].ptr,
                self.output_modes,
                self.compute_type,
            )

        # ========================================================================
        # Query scalar type and initialize planning/workspace state
        # ========================================================================
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

        # Initialize planning-related members
        self.contraction_planned = False
        self.plan_preference_ptr = cutensor.create_plan_preference(self.handle, cutensor.Algo.DEFAULT, cutensor.JitMode.NONE)
        self._plan_preference = ContractionPlanPreference(self)

        # Initialize remaining members
        self.workspace_ptr = None
        self.workspace_allocated_size = 0
        self.workspace_size = None
        self.workspace_stream = None
        self.workspace_allocated_here = False
        self.last_compute_event = None
        self.plan_ptr = None

        if log_info:
            self.logger.info(f"The {self.__class__.__name__} object has been created.")

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
        if self._operands_released:
            raise RuntimeError(
                f"{what} cannot be performed after the operands have been released. "
                f"Use reset_operands() to provide new operands before performing the {what.lower()}."
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
        log_debug = self.logger.isEnabledFor(logging.DEBUG)
        assert self.workspace_size is not None, "Internal Error."
        assert self.workspace_allocated_here is False, "Internal Error."

        if self.workspace_size == 0:  # For performance, bypass allocator for workspace size == 0.
            self.workspace_ptr = memory.MemoryPointer(0, 0, finalizer=None)
        else:
            if log_debug:
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
        if log_debug:
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
    def reset_operands(self, *, a=None, b=None, c=None, d=None, out=None, stream=None):
        if self.num_inputs == 2 and d is not None:
            raise RuntimeError("Internal Error: For pairwise contractions, d can not be set.")

        # if the operands have been released, all operands must be provided
        if self._operands_released:
            all_provided = (
                a is not None
                and b is not None
                and (c is not None or not self.c_provided)
                and (d is not None or not self.d_provided)
                and (out is not None or not self.output_provided)
            )
            if not all_provided:
                raise ValueError("After release_operands(), all operands must be provided.")

        stream_holder = None  # lazy initialization
        for op_name, op in zip(["a", "b", "c", "d", "out"], [a, b, c, d, out], strict=False):
            # if op is None, we keep the original value of the operand,
            # so skip the rest of the logic for this operand
            if op is None:
                continue

            if (op_name == "c" and not self.c_provided) or (op_name == "d" and not self.d_provided):
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

            tensor_desc = self.tensor_descs[op_name]

            error_pattern = (
                f"The operand {op_name} must have the same {{attr}} "
                f"as the one specified during the initialization of the "
                f"ElementaryContraction object."
            )

            if tensor_desc.shape != op.shape:
                raise ValueError(error_pattern.format(attr="shape"))

            if tensor_desc.dtype != op.dtype:
                raise ValueError(error_pattern.format(attr="dtype"))

            if tensor_desc.strides != op.strides:
                raise ValueError(error_pattern.format(attr="strides"))

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
            elif cutensor_utils.compute_pointer_alignment(op.data_ptr) % self.tensor_descs[op_name].alignment:
                # If the operand is in the same memory space as the
                # execution space (copy not needed), we need to check if the pointer
                # alignment is the same or a multiple of the original pointer alignments.
                # If the operand is in host memory, this check can be skipped as we
                # internally copy the operand to the execution space and
                # device ptr alignment should be guaranteed to be the same.
                raise ValueError(
                    f"The pointer alignment of the operand {op_name} must be the same or a multiple of the corresponding "
                    f"pointer alignment specified during the initialization of the {self.__class__.__name__} object."
                )

            setattr(self, op_name, op)
            self.logger.info(f"operand {op_name} has been reset to the new operand provided.")
        self._operands_released = False
        return

    @utils.precondition(_check_valid_contraction)
    def _release_operands(self):
        # We release the internal tensor references held by the wrappers
        # for the user-provided operands and/or their GPU mirrors.
        # The TensorHolder wrappers themselves are kept alive so that
        # reset_operands_unchecked can reuse them without re-wrapping.
        self.a.tensor = None
        self.b.tensor = None
        if self.c_provided:
            self.c.tensor = None
        if self.d_provided:
            self.d.tensor = None
        if self.output_provided:
            # self.out_return always holds the user's tensor.
            # self.out holds the user's tensor when operand memory space
            # matches execution (same device; same object as self.out_return),
            # or an internal device mirror when they differ (inputs on CPU).
            # In both cases, release the inner tensor references.
            self.out_return.tensor = None
            self.out.tensor = None
        self._operands_released = True
        self.logger.info("User-provided operands have been released.")

    def _reset_operands_unchecked(
        self, *, a=None, b=None, c=None, d=None, out=None, stream: utils.AnyStream | int | None = None
    ):
        # Since we have the caller's compatibility guarantee for the inputs, we can leverage
        # the metadata stored during initialization to know if the inputs were on the GPU.
        # If the memory space of the inputs matches the execution space, namely the inputs
        # during initialization resided on the GPU, then we know/expect the newly provided
        # inputs must also reside on the GPU. The TensorHolder wrappers are always alive
        # (release_operands only clears .tensor), so we can unconditionally swap the inner
        # tensor — no wrap_operand() needed.
        if self.input_device_id != "cpu":
            if a is not None:
                self.a.tensor = a
            if b is not None:
                self.b.tensor = b
            if c is not None:
                self.c.tensor = c
            if d is not None:
                self.d.tensor = d
            if out is not None:
                self.out.tensor = out
                self.out_return.tensor = out

            self._operands_released = False
            return

        # if we are here, it means the inputs were on the CPU
        # so we need to distinguish between the case where the operands
        # were released and the case where they were not released.
        if self._operands_released:
            # CPU inputs after release: device mirrors were freed, need re-allocation.
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
            if a is not None:
                self.a = tensor_wrapper.wrap_operand(a).to(self.execution_device_id, stream_holder)
            if b is not None:
                self.b = tensor_wrapper.wrap_operand(b).to(self.execution_device_id, stream_holder)
            if c is not None:
                self.c = tensor_wrapper.wrap_operand(c).to(self.execution_device_id, stream_holder)
            if d is not None:
                self.d = tensor_wrapper.wrap_operand(d).to(self.execution_device_id, stream_holder)
            if out is not None:
                out_wrapped = tensor_wrapper.wrap_operand(out)
                self.out = out_wrapped.to(self.execution_device_id, stream_holder)
                self.out_return = out_wrapped
        else:
            # CPU inputs, not released: device mirrors are valid, copy into them.
            stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)
            if a is not None:
                self.a.copy_(tensor_wrapper.wrap_operand(a), stream_holder=stream_holder)
            if b is not None:
                self.b.copy_(tensor_wrapper.wrap_operand(b), stream_holder=stream_holder)
            if c is not None:
                self.c.copy_(tensor_wrapper.wrap_operand(c), stream_holder=stream_holder)
            if d is not None:
                self.d.copy_(tensor_wrapper.wrap_operand(d), stream_holder=stream_holder)
            if out is not None:
                out_wrapped = tensor_wrapper.wrap_operand(out)
                self.out.copy_(out_wrapped, stream_holder=stream_holder)
                self.out_return = out_wrapped

        self._operands_released = False

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
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        # A new plan needs to be created at each plan() call
        if self.plan_ptr is not None:
            cutensor.destroy_plan(self.plan_ptr)
            self.plan_ptr = None
            if log_debug:
                self.logger.debug("The previous plan has been destroyed.")

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
        if log_info:
            self.logger.info(f"The required workspace size for the contraction operation is {self.workspace_size}.")
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
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        self.alpha[0] = alpha
        self.beta[0] = beta

        stream_holder = utils.get_or_create_stream(self.execution_device_id, stream, self.internal_package)

        # The buffer to be used for cutensor execution
        # If out was provided during initialization, this would have been created
        # regardless of where it resided (CPU or GPU).
        # If out was not provided during initialization, we need to create a new buffer
        if self.out is None:
            assert not self.output_provided, (
                "Internal Error: out cannot be None if the output was provided during initialization."
            )
            self.out = utils.create_empty_tensor(
                self.a.__class__, self.output_shape, self.data_type, self.execution_device_id, stream_holder, False
            )
            if log_debug:
                self.logger.debug(
                    f"The output tensor of type {type(self.out.tensor)} with shape {self.out.shape} has been created."
                )

        if log_info:
            self.logger.info("= EXECUTION PHASE =")
            self.logger.info("Starting tensor contraction calculation...")
            self.logger.info(f"{self.call_prologue}")
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

        if self.output_provided:
            if self.execution_device_id != self.input_device_id:
                self.out_return.copy_(self.out, stream_holder)
            return self.out_return.tensor
        else:
            if self.execution_device_id != self.input_device_id:
                return self.out.to(self.input_device_id, stream_holder).tensor
            else:
                out = self.out
                # release the output tensor as the ownership is transferred to the caller
                self.out = None
                return out.tensor

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

        class_name = self.__class__.__name__
        try:
            if self.last_compute_event is not None and self.workspace_stream is not None:
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            self._free_plan_resources()

            # Free handle if we own it.
            if self.handle is not None and self.own_handle:
                cutensor.destroy(self.handle)
                self.handle, self.own_handle = None, False

            if self.contraction_desc is not None:
                cutensor.destroy_operation_descriptor(self.contraction_desc)
                self.contraction_desc = None

            self.tensor_descs = None

            # Set all attributes to None except for logger and valid_state
            _keep = {"logger", "valid_state"}
            for attr in list(vars(self)):
                if attr not in _keep:
                    setattr(self, attr, None)

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
        :attr:`plan_preference`, :meth:`plan`, :meth:`reset_operands`,
        :meth:`release_operands`, :meth:`execute`

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
        # Check here for a valid expr because it is a precondition for the constructor
        # of the base class where it is used to extract the number of operands.
        if not isinstance(expr, str) or expr.count(",") != 1:
            raise ValueError("Binary contraction requires a string with exactly 2 comma-separated operands")

        super().__init__(expr, a, b, c=c, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution)

    def reset_operands(self, *, a=None, b=None, c=None, out=None, stream=None):
        """
        Reset one or more operands held by this :class:`BinaryContraction` instance.
        Only the operands explicitly passed are updated; omitted operands retain
        their current values.

        .. versionchanged:: 0.9
            All parameters are now keyword-only.

        Args:
            a: {a}

            b: {b}

            c: {addend}

            out: {out}

            stream: {stream}

        {reset_operands_semantics}

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

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is
            equivalent to updating the operands in-place, i.e, replacing
            ``contraction.reset_operands(a=a1, b=b1)`` with ``a[:]=a1`` and ``b[:]=b1``.
            Note that updating the operand in-place should be adopted with caution as it can
            only yield the expected result under the additional constraint below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction/example06_stateful_inplace.py>`_.

        .. seealso::
            :meth:`reset_operands_unchecked`, :meth:`release_operands`
        """
        super().reset_operands(a=a, b=b, c=c, out=out, stream=stream)

    def reset_operands_unchecked(self, *, a=None, b=None, c=None, out=None, stream: utils.AnyStream | int | None = None):
        """
        {reset_operands_unchecked}
        """
        super()._reset_operands_unchecked(a=a, b=b, c=c, out=out, stream=stream)

    def release_operands(self):
        """
        {release_operands}
        """
        self._release_operands()


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
        :attr:`plan_preference`, :meth:`plan`, :meth:`reset_operands`,
        :meth:`release_operands`, :meth:`execute`

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
        # Check here for a valid expr because it is a precondition for the constructor
        # of the base class where it is used to extract the number of operands.
        if not isinstance(expr, str) or expr.count(",") != 2:
            raise ValueError("Ternary contraction requires a string with exactly 3 comma-separated operands")

        super().__init__(
            expr, a, b, c=c, d=d, out=out, qualifiers=qualifiers, stream=stream, options=options, execution=execution
        )

    def reset_operands(self, *, a=None, b=None, c=None, d=None, out=None, stream=None):
        """
        Reset one or more operands held by this :class:`TernaryContraction` instance.
        Only the operands explicitly passed are updated; omitted operands retain
        their current values.

        .. versionchanged:: 0.9
            All parameters are now keyword-only.

        Args:
            a: {a}

            b: {b}

            c: {c}

            d: {addend}

            out: {out}

            stream: {stream}

        {reset_operands_semantics}

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

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands`
            is equivalent to updating the operands in-place, i.e, replacing
            ``contraction.reset_operands(a=a1, b=b1, c=c1)`` with ``a[:]=a1``
            and ``b[:]=b1`` and ``c[:]=c1``. Note that updating the operand in-place
            should be adopted with caution as it can only yield the expected result
            under the additional constraint below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction/example06_stateful_inplace.py>`_.

        .. seealso::
            :meth:`reset_operands_unchecked`, :meth:`release_operands`
        """
        super().reset_operands(a=a, b=b, c=c, d=d, out=out, stream=stream)

    def reset_operands_unchecked(
        self, *, a=None, b=None, c=None, d=None, out=None, stream: utils.AnyStream | int | None = None
    ):
        """
        {reset_operands_unchecked}
        """
        super()._reset_operands_unchecked(a=a, b=b, c=c, d=d, out=out, stream=stream)

    def release_operands(self):
        """
        {release_operands}
        """
        self._release_operands()


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
        :class:`ContractionPlanPreference`

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
        contraction.plan(stream=stream)
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
        :class:`ContractionPlanPreference`

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
        contraction.plan(stream=stream)
        out = contraction.execute(alpha=alpha, beta=beta, stream=stream)
    return out
