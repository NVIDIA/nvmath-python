.. module:: nvmath.bindings.nvpl.fft

NVPL FFT (:mod:`nvmath.bindings.nvpl.fft`)
==========================================

For detailed documentation on the original C APIs, refer to the `NVPL FFT documentation
<https://docs.nvidia.com/nvpl/latest/fft/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   FFTWError
   FFTWUnaligned
   Kind
   Plan
   PlannerFlags
   Precision
   Sign


Functions
*********

.. autosummary::
   :toctree: generated/

   cleanup_threads
   cleanup_threads_double
   cleanup_threads_float
   destroy
   destroy_plan_double
   destroy_plan_float
   execute
   execute_c2c_double
   execute_c2c_float
   execute_c2r_double
   execute_c2r_float
   execute_r2c_double
   execute_r2c_float
   get_version
   init_threads
   init_threads_double
   init_threads_float
   plan_many
   plan_many_c2c_double
   plan_many_c2c_float
   plan_many_c2r_double
   plan_many_c2r_float
   plan_many_r2c_double
   plan_many_r2c_float
   plan_with_nthreads
   plan_with_nthreads_double
   plan_with_nthreads_float
   planner_nthreads
   planner_nthreads_double
   planner_nthreads_float
