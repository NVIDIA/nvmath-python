.. module:: nvmath.bindings.cublasMp

cuBLASMp (:mod:`nvmath.bindings.cublasMp`)
==========================================

For detailed documentation on the original C APIs, refer to the `cuBLASMp documentation
<https://docs.nvidia.com/cuda/cublasmp>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   ComputeType
   cuBLASMpError
   GridLayout
   MatmulAlgoType
   MatmulDescriptorAttribute
   MatmulEpilogue
   MatmulMatrixScale
   Operation
   Status

Functions
*********

.. autosummary::
   :toctree: generated/

   create
   destroy
   stream_set
   get_version
   grid_create
   grid_destroy
   matrix_descriptor_create
   matrix_descriptor_destroy
   matmul_descriptor_create
   matmul_descriptor_destroy
   matmul_descriptor_attribute_set
   matmul_descriptor_attribute_get
   matmul_buffer_size
   matmul
   numroc
