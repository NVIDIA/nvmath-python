.. module:: nvmath.bindings.curand

cuRAND (:mod:`nvmath.bindings.curand`)
======================================

For detailed documentation on the original C APIs, refer to the `cuRAND documentation
<https://docs.nvidia.com/cuda/curand/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   RngType
   Ordering
   Method
   Status
   cuRANDError

Functions
*********

.. autosummary::
   :toctree: generated/

   create_generator
   create_generator_host
   destroy_generator
   get_version
   get_property
   set_stream
   set_pseudo_random_generator_seed
   set_generator_offset
   set_generator_ordering
   set_quasi_random_generator_dimensions
   generate
   generate_long_long
   generate_uniform
   generate_uniform_double
   generate_normal
   generate_normal_double
   generate_log_normal
   generate_log_normal_double
   create_poisson_distribution
   destroy_distribution
   generate_poisson
   generate_poisson_method
   generate_binomial
   generate_binomial_method
   generate_seeds
