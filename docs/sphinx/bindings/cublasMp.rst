.. module:: nvmath.bindings.cublasMp

cuBLASMp (:mod:`nvmath.bindings.cublasMp`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuBLASMp documentation
<https://docs.nvidia.com/cuda/cublasmp>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   EmulationStrategy
   GridLayout
   MatmulAlgoType
   MatmulDescriptorAttribute
   MatmulEpilogue
   MatmulMatrixScale
   Status
   cuBLASMpError

Functions
*********

.. autosummary::
   :toctree: generated/

   create
   destroy
   stream_set
   stream_get
   get_version
   set_emulation_strategy
   get_emulation_strategy
   grid_create
   grid_destroy
   matrix_descriptor_create
   matrix_descriptor_init
   matrix_descriptor_destroy
   numroc
   matmul_descriptor_create
   matmul_descriptor_init
   matmul_descriptor_destroy
   matmul_descriptor_attribute_set
   matmul_descriptor_attribute_get
   matmul_buffer_size
   matmul
   geadd_buffer_size
   geadd
   gemm_buffer_size
   gemm
   gemr2d_buffer_size
   gemr2d
   syrk_buffer_size
   syrk
   tradd_buffer_size
   tradd
   trmr2d_buffer_size
   trmr2d
   trsm_buffer_size
   trsm
