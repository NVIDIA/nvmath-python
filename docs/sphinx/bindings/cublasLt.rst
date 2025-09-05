.. module:: nvmath.bindings.cublasLt

cuBLASLt (:mod:`nvmath.bindings.cublaslt`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuBLASLt documentation
<https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   BatchMode
   ClusterShape
   cuBLASLtError
   Epilogue
   MatmulAlgo
   MatmulAlgoCapAttribute
   MatmulAlgoConfigAttribute
   MatmulDescAttribute
   MatmulHeuristicResult
   MatmulInnerShape
   MatmulMatrixScale
   MatmulPreferenceAttribute
   MatmulSearch
   MatmulStages
   MatmulTile
   MatrixLayoutAttribute
   MatrixTransformDescAttribute
   Order
   PointerMode
   PointerModeMask
   ReductionScheme


Functions
*********

.. autosummary::
   :toctree: generated/

   create
   destroy
   disable_cpu_instructions_set_mask
   get_cudart_version
   get_matmul_algo_cap_attribute_dtype
   get_matmul_algo_config_attribute_dtype
   get_matmul_desc_attribute_dtype
   get_matmul_preference_attribute_dtype
   get_matrix_layout_attribute_dtype
   get_matrix_transform_desc_attribute_dtype
   get_property
   get_status_name
   get_status_string
   get_version
   heuristics_cache_get_capacity
   heuristics_cache_set_capacity
   logger_force_disable
   logger_open_file
   logger_set_level
   logger_set_mask
   matmul
   matmul_algo_cap_get_attribute
   matmul_algo_check
   matmul_algo_config_get_attribute
   matmul_algo_config_set_attribute
   matmul_algo_get_heuristic
   matmul_algo_get_ids
   matmul_algo_init
   matmul_desc_create
   matmul_desc_destroy
   matmul_desc_get_attribute
   matmul_desc_set_attribute
   matmul_preference_create
   matmul_preference_destroy
   matmul_preference_get_attribute
   matmul_preference_set_attribute
   matrix_layout_create
   matrix_layout_destroy
   matrix_layout_get_attribute
   matrix_layout_set_attribute
   matrix_transform
   matrix_transform_desc_create
   matrix_transform_desc_destroy
   matrix_transform_desc_get_attribute
   matrix_transform_desc_set_attribute
