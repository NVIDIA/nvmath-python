*********************
Sparse Linear Algebra
*********************

.. _sparse-overview:

Overview
========

Sparse tensors are vectors, matrices, and higher-dimensional generalizations with many
zeros.  Such tensors are crucial in various fields such as scientific computing, signal
processing, and deep learning due to their efficiency in storage, computation, and power.
Sparse linear algebra refers to any linear algebra where vectors or matrices are sparse.
Higher-dimensional sparse tensors may occur in deep learning.

The sparse linear algebra module :mod:`nvmath.sparse` in nvmath-python leverages various
NVIDIA math libraries to support sparse [#]_ linear algebra computations. As of the current
Beta release, we offer the Universal Sparse Tensor (UST) to simplify managing sparsity as
well as specialized sparse direct solver API based on the `cuDSS library
<https://docs.nvidia.com/cuda/cudss/>`_. Distributed (multi-node multi-GPU) execution is not
currently supported.

.. _ust-description:

Universal Sparse Tensor
=======================

The Universal Sparse Tensor (UST) simplifies managing sparsity by decoupling a
tensor's inherent sparsity from its memory storage representation. With roots
in sparse compiler technology (see e.g. `MT1`_ or `TACO`_), the UST uses a
**tensor format DSL** (Domain Specific Language) to describe how the sparse
tensor should be represented and uses **type polymorphism** on a small set of
base operations to define the vast space of instances for these operations.
Developers merely focus on the sparsity of a tensor. Runtime inspection
of the eventually chosen format decides whether to dispatch to a heavily optimized
library or kernel, or to fill in the “holes” with automatic sparse code
generation when no such solution exists yet.

The nvmath-python UST implementation provides interoperability with tensors of
PyTorch, SciPy, CuPy, and NumPy. The interoperability is zero-cost, which means
that viewing dense or sparse formats like COO, CSR, CSC, BSR, BSC, and DIA as
a UST object or back is done without data movement or copying. Instead, the UST
object references the storage buffers of the original data structure.

.. _MT1:  https://aartbik.com/sparse.php
.. _TACO: http://tensor-compiler.org/

.. _ust-dsl:

UST DSL
-------

The tensor format DSL of the Universal Sparse Tensor maps tensor dimensions
(logical tensor indices) to storage levels (physical memory indices) using
an invertible function that defines how each level should be stored.

The DSL consists of:

*  An ordered sequence of dimension specifications, each of which includes:

   *  a **dimension-expression**, which provides a reference to each dimension

*  An ordered sequence of level specifications, each of which includes:

   *  a **level expression**, which defines what is stored in each level
   *  a **level type**, which defines how the level is stored, including:

      *  a required **level format**
      *  a collection of **level properties**

The following level formats are supported:

*  **dense**: the level is dense, entries along the level are stored linearized
   without any metadata but indexing logic only
*  **batch**: a variant of the dense format, indicating that any subsequent
   compression is not linearized
*  **range**: a variant of the dense format, restricting the range based on a
   compression expression in the previous level
*  **compressed**: the level is sparse, only nonzeros along the level are stored
   with positions and coordinates in two arrays at that level
*  **singleton**: a variant of the compressed format, for when coordinates have no
   siblings which means that the positions array can be omitted at that level
*  **delta(b)**: a variant of the compressed format, where coordinates denote
   the distance to the next stored entry; zero padding is used when this
   distance exceeds b-bits

Level formats have, at least, the following level properties:

* **non/unique** : duplicates are (not) allowed at that level (unique by default)
* **un/ordered** : coordinates (not) sorted at that level (ordered by default)

The UST type can easily define many common storage formats (such as dense
vectors and matrices, sparse vectors, sparse matrices in formats like COO, CSR,
CSC, DCSR, DCSC, BSR, BSC, DIA, and with generalizations to sparse tensors),
as well as many less common and rather novel storage formats, as was demonstrated
in this `UST blog posting`_.

In nvmath-python, we adopt a Pythonic way of presenting the DSL, trading some
inspection performance for extreme runtime flexibility.
The CSC format, for instance, is expressed as follows::

    CSC = TensorFormat( [i, j], {j: LevelFormat.DENSE, i: LevelFormat.COMPRESSED} )

A major advantage of such objects is that everything can be constructed dynamically
at runtime (including parsing from strings). Performing format-specific tasks, however,
requires inspecting the actual contents at runtime. Since such decisions generally
happen outside performance-critical paths, trading off for generality seems an
acceptable choice.

.. _UST blog posting: https://developer.nvidia.com/blog/establishing-a-scalable-sparse-ecosystem-with-the-universal-sparse-tensor/

.. _ust-grammar:

UST DSL Grammar
---------------

The grammar of the tensor format DSL in `Backus-Naur Form`_ is as follows::

  <tensor_format> ::= ( <dim_specs> ) -> ( <lvl_specs> )

  <dim_specs> ::= <empty>  | <dim_list>
  <dim_list>  ::= <dim_spec> | <dim_spec> , <dim_list>

  <dim_spec>  ::= <dim_expr>
  <dim_expr>  ::= <dim_var>  # only simple for now
  <dim_var>   ::= <id>       # e.g. i or d0

  <lvl_specs> ::= <empty>  | <lvl_list>
  <lvl_list>  ::= <lvl_spec> | <lvl_spec> , <lvl_list>

  <lvl_spec>  ::= <lvl_expr> : <lvl_type>

  <lvl_expr>  ::= <lvl_expr> + <lvl_expr>    | <lvl_expr> - <lvl_expr>  |
                  <lvl_expr> / <lvl_expr>    | <lvl_expr> % <lvl_expr>  |
                  <lvl_expr> count <lvl_expr> |
                  <dim_var>                   |
                  <const>  # e.g. 4 or 200

  <lvl_type>   ::= <lvl_format> | <lvl_format> ( <lvl_props> )
  <lvl_format> ::= dense | compressed | singleton | range | delta(<const>)
  <lvl_props>  ::= <lvl_prop> | <lvl_prop> , <lvl_props>
  <lvl_prop>   ::= ordered | unordered | unique | nonunique

.. _Backus-Naur Form: https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form

Even though the grammar allows for many formats that are syntactically correct,
only specific grammar instances actually make sense. For example, dimension variables
should only appear once in the dimension specification (e.g. ``(i,i) -> ...`` is
invalid), level expressions are either a single dimension variable (like ``i``), or
a non-nested add/sub operation of two such variables (like ``i-j``) or a div/mod
operation with a constant (like ``i/4``). Also, the mapping should always remain an
invertible, one-to-one function between dimensions and levels. Lastly, not all property
combinations make sense. For example, dense formats cannot be nonunique or unordered. The
UST constructor gives an error for tensor formats that violate these constraints.

.. note::
   The grammar can be expanded in the future to further generalize the UST to include
   storage formats that cannot be expressed currently.

.. _sparse-api-reference:

API Reference
=============

.. module:: nvmath.sparse.ust

Universal Sparse Tensor (UST) APIs (:mod:`nvmath.sparse.ust`)
-------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   Tensor
   Dimension
   LevelFormat
   LevelProperty
   LevelExpr
   TensorFormat
   NamedFormats

.. module:: nvmath.sparse.ust.interfaces.torch_interface

UST Torch Interface (:mod:`nvmath.sparse.ust.interfaces.torch_interface`)
-------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :template: torchust

   TorchUST

.. autosummary::
   :toctree: generated/
   :template: function

   prune_model
   reformat_model

.. module:: nvmath.sparse

Generic Linear Algebra APIs (:mod:`nvmath.sparse`)
--------------------------------------------------

.. autosummary::
   :toctree: generated/

   matmul_matrix_qualifiers_dtype
   compile_matmul_prolog
   compile_matmul_epilog
   compile_matmul_add
   compile_matmul_atomic_add
   compile_matmul_mul
   Matmul
   matmul
   ComputeType

.. autosummary::
   :toctree: generated/
   :template: dataclass

   ExecutionCUDA
   MatmulOptions

.. module:: nvmath.sparse.advanced

Specialized Linear Algebra APIs (:mod:`nvmath.sparse.advanced`)
---------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   direct_solver
   DirectSolver
   DirectSolverFactorizationConfig
   DirectSolverFactorizationInfo
   DirectSolverPlanConfig
   DirectSolverPlanInfo
   DirectSolverSolutionConfig
   memory_estimates_dtype

.. autosummary::
   :toctree: generated/
   :template: dataclass

   DirectSolverAlgType
   DirectSolverMatrixType
   DirectSolverMatrixViewType
   DirectSolverOptions
   ExecutionCUDA
   ExecutionHybrid
   HybridMemoryModeOptions
   DirectSolverPlanPreferences
   DirectSolverFactorizationPreferences
   DirectSolverSolutionPreferences

.. rubric:: Footnotes

.. [#] See :ref:`Linear Algebra <linalg-overview>` for dense operations.
