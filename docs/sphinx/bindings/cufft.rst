.. module:: nvmath.bindings.cufft

cuFFT (:mod:`nvmath.bindings.cufft`)
====================================

For detailed documentation on the original C APIs, refer to the `cuFFT documentation
<https://docs.nvidia.com/cuda/cufft/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   Compatibility
   cuFFTError
   LibFormat
   Property
   Result
   Type
   XtCallbackType
   XtCopyType
   XtQueryType
   XtSubFormat
   XtWorkAreaPolicy

Functions
*********

.. autosummary::
   :toctree: generated/

   check_status
   create
   destroy
   estimate_many
   estimate1d
   estimate2d
   estimate3d
   exec_c2c
   exec_c2r
   exec_d2z
   exec_r2c
   exec_z2d
   exec_z2z
   get_plan_property_int64
   get_property
   get_size
   get_size_many
   get_size_many64
   get_size1d
   get_size2d
   get_size3d
   get_version
   make_plan_many
   make_plan_many64
   make_plan1d
   make_plan2d
   make_plan3d
   plan_many
   plan1d
   plan2d
   plan3d
   reset_plan_property
   set_auto_allocation
   set_plan_property_int64
   set_stream
   set_work_area
   xt_clear_callback
   xt_exec
   xt_exec_descriptor
   xt_exec_descriptor_c2c
   xt_exec_descriptor_c2r
   xt_exec_descriptor_d2z
   xt_exec_descriptor_r2c
   xt_exec_descriptor_z2d
   xt_exec_descriptor_z2z
   xt_free
   xt_get_size_many
   xt_make_plan_many
   xt_malloc
   xt_memcpy
   xt_query_plan
   xt_set_callback_shared_size
   xt_set_gpus
   xt_set_jit_callback
   xt_set_subformat_default
   xt_set_work_area
   xt_set_work_area_policy
