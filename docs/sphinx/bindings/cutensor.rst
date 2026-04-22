.. module:: nvmath.bindings.cutensor

cuTENSOR (:mod:`nvmath.bindings.cutensor`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuTENSOR documentation
<https://docs.nvidia.com/cuda/cutensor/latest/index.html>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   ComputeDesc
   cuTENSORError
   Operator
   Status
   Algo
   WorksizePreference
   OperationDescriptorAttribute
   PlanPreferenceAttribute
   AutotuneMode
   JitMode
   CacheMode
   PlanAttribute

Functions
*********

.. autosummary::
   :toctree: generated/

   create
   destroy
   handle_resize_plan_cache
   handle_write_plan_cache_to_file
   handle_read_plan_cache_from_file
   write_kernel_cache_to_file
   read_kernel_cache_from_file
   create_tensor_descriptor
   destroy_tensor_descriptor
   create_elementwise_trinary
   elementwise_trinary_execute
   create_elementwise_binary
   elementwise_binary_execute
   create_permutation
   permute
   create_contraction
   destroy_operation_descriptor
   get_operation_descriptor_attribute_dtype
   operation_descriptor_set_attribute
   operation_descriptor_get_attribute
   create_plan_preference
   destroy_plan_preference
   get_plan_preference_attribute_dtype
   plan_preference_set_attribute
   plan_preference_get_attribute
   get_plan_attribute_dtype
   plan_get_attribute
   estimate_workspace_size
   create_plan
   destroy_plan
   contract
   create_reduction
   reduce
   create_contraction_trinary
   contract_trinary
   create_block_sparse_tensor_descriptor
   destroy_block_sparse_tensor_descriptor
   create_block_sparse_contraction
   block_sparse_contract
   get_error_string
   get_version
   get_cudart_version
   logger_set_file
   logger_open_file
   logger_set_level
   logger_set_mask
   logger_force_disable
