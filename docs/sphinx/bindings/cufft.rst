.. module:: nvmath.bindings.cufft

cuFFT (:mod:`nvmath.bindings.cufft`)
====================================

For detailed documentation on the original C APIs, please refer to `cuFFT documentation <https://docs.nvidia.com/cuda/cufft/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   LibFormat
   Result
   Type
   Compatibility
   XtSubFormat
   XtCopyType
   XtQueryType
   XtWorkAreaPolicy
   XtCallbackType
   Property
   cuFFTError

Functions
*********

.. autosummary::
   :toctree: generated/

   plan1d
   plan2d
   plan3d
   plan_many
   make_plan1d
   make_plan2d
   make_plan3d
   make_plan_many
   make_plan_many64
   get_size_many64
   estimate1d
   estimate2d
   estimate3d
   estimate_many
   create
   get_size1d
   get_size2d
   get_size3d
   get_size_many
   get_size
   set_work_area
   set_auto_allocation
   exec_c2c
   exec_r2c
   exec_c2r
   exec_z2z
   exec_d2z
   exec_z2d
   set_stream
   destroy
   get_version
   get_property
   xt_set_gpus
   xt_malloc
   xt_memcpy
   xt_free
   xt_set_work_area
   xt_exec_descriptor_c2c
   xt_exec_descriptor_r2c
   xt_exec_descriptor_c2r
   xt_exec_descriptor_z2z
   xt_exec_descriptor_d2z
   xt_exec_descriptor_z2d
   xt_query_plan
   xt_clear_callback
   xt_set_callback_shared_size
   xt_make_plan_many
   xt_get_size_many
   xt_exec
   xt_exec_descriptor
   xt_set_work_area_policy
   xt_set_jit_callback
   xt_set_subformat_default
