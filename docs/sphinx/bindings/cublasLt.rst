.. module:: nvmath.bindings.cublasLt

cuBLASLt (:mod:`nvmath.bindings.cublaslt`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuBLASLt documentation
<https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   MatmulTile
   MatmulStages
   PointerMode
   PointerModeMask
   Order
   MatrixLayoutAttribute
   MatmulDescAttribute
   MatrixTransformDescAttribute
   ReductionScheme
   Epilogue
   MatmulSearch
   MatmulPreferenceAttribute
   MatmulAlgoCapAttribute
   MatmulAlgoConfigAttribute
   ClusterShape
   MatmulInnerShape
   cuBLASLtError


Functions
*********

.. autosummary::
   :toctree: generated/

   create
   destroy
   get_version
   get_cudart_version
   get_property
   matmul
   matrix_transform
   matrix_layout_create
   matrix_layout_destroy
   get_matrix_layout_attribute_dtype
   matrix_layout_set_attribute
   matrix_layout_get_attribute
   matmul_desc_create
   matmul_desc_destroy
   get_matmul_desc_attribute_dtype
   matmul_desc_set_attribute
   matmul_desc_get_attribute
   matrix_transform_desc_create
   matrix_transform_desc_destroy
   get_matrix_transform_desc_attribute_dtype
   matrix_transform_desc_set_attribute
   matrix_transform_desc_get_attribute
   matmul_preference_create
   matmul_preference_destroy
   get_matmul_preference_attribute_dtype
   matmul_preference_set_attribute
   matmul_preference_get_attribute
   matmul_algo_get_heuristic
   matmul_algo_init
   matmul_algo_check
   get_matmul_algo_cap_attribute_dtype
   matmul_algo_cap_get_attribute
   get_matmul_algo_config_attribute_dtype
   matmul_algo_config_set_attribute
   matmul_algo_config_get_attribute
   logger_open_file
   logger_set_level
   logger_set_mask
   logger_force_disable
   get_status_name
   get_status_string
   heuristics_cache_get_capacity
   heuristics_cache_set_capacity
   disable_cpu_instructions_set_mask
   matmul_algo_get_ids
