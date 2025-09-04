.. module:: nvmath.bindings.curand

cuRAND (:mod:`nvmath.bindings.curand`)
======================================

For detailed documentation on the original C APIs, refer to the `cuRAND documentation
<https://docs.nvidia.com/cuda/curand/>`_.

Enums and constants
*******************

.. autosummary::
   :toctree: generated/

   cuRANDError
   DirectionVectorSet
   Method
   Ordering
   RngType
   Status

Functions
*********

.. autosummary::
   :toctree: generated/

   check_status
   create_generator
   create_generator_host
   create_poisson_distribution
   destroy_distribution
   destroy_generator
   generate
   generate_binomial
   generate_binomial_method
   generate_log_normal
   generate_log_normal_double
   generate_long_long
   generate_normal
   generate_normal_double
   generate_poisson
   generate_poisson_method
   generate_seeds
   generate_uniform
   generate_uniform_double
   get_direction_vectors32
   get_direction_vectors64
   get_property
   get_scramble_constants32
   get_scramble_constants64
   get_version
   set_generator_offset
   set_generator_ordering
   set_pseudo_random_generator_seed
   set_quasi_random_generator_dimensions
   set_stream
