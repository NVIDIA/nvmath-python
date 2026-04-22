# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
The tensor format DSL of the Universal Sparse Tensor maps tensor dimensions
(logical tensor indices) to storage levels (physical memory indices) using
an invertible function that defines how each level should be stored.
"""

__all__ = [
    "is_ordered",
    "is_unique",
    "Dimension",
    "LevelFormat",
    "LevelProperty",
    "LevelExpr",
    "TensorFormat",
    "NamedFormats",
]

from collections.abc import Sequence
from enum import IntEnum


def is_ordered(prop):
    """Determines if given property is ordered."""
    if isinstance(prop, Sequence):
        for p in prop:
            if not is_ordered(p):
                return False
    elif isinstance(prop, LevelProperty) and prop == LevelProperty.UNORDERED:
        return False
    return True  # ordered unless seen otherwise


def is_unique(prop):
    """Determines if given property is unique."""
    if isinstance(prop, Sequence):
        for p in prop:
            if not is_unique(p):
                return False
    elif isinstance(prop, LevelProperty) and prop == LevelProperty.NONUNIQUE:
        return False
    return True  # unique unless seen otherwise


class LevelFormat(IntEnum):
    DENSE = 0
    BATCH = 1
    COMPRESSED = 2
    SINGLETON = 3
    RANGE = 4
    DELTA = 5

    def __repr__(self):
        """Returns representation without index."""
        return f"<{self.__class__.__name__}.{self.name}>"


class LevelProperty(IntEnum):
    UNIQUE = 0
    NONUNIQUE = 1
    ORDERED = 2
    UNORDERED = 3

    def __repr__(self):
        """Returns representation without index."""
        return f"<{self.__class__.__name__}.{self.name}>"


# The inverse map is used to check for certain incompatible property sequences.
LevelPropertyInverse = {
    LevelProperty.UNIQUE: LevelProperty.NONUNIQUE,
    LevelProperty.NONUNIQUE: LevelProperty.UNIQUE,
    LevelProperty.ORDERED: LevelProperty.UNORDERED,
    LevelProperty.UNORDERED: LevelProperty.ORDERED,
}


def _find_range(levels, expr1, expr2):
    for rl, (k, v) in enumerate(levels.items()):
        if v == LevelFormat.RANGE and k in (expr1, expr2):
            return rl, k
    raise ValueError(f"Cannot find {expr1} or {expr2} in levels")


class Operator:
    """
    An abstract binary level expression operator type.
    """

    # TODO: use protocol.
    pass


class Add(Operator):
    """
    Add two level expressions.
    """

    def __call__(self, expr1, expr2, as_size):
        if as_size:
            return expr1 + expr2 - 1
        return expr1 + expr2

    def invert(self, expr1, expr2, dimensions, levels, dim_indices, lvl_indices, idx):
        rl, re = _find_range(levels, expr1, expr2)
        if re == expr2:
            dim_indices[dimensions.index(expr1)] = lvl_indices[idx] - lvl_indices[rl]
        else:
            assert re == expr1
            dim_indices[dimensions.index(expr2)] = lvl_indices[idx] - lvl_indices[rl]

    def __repr__(self):
        return "+"


class Subtract(Operator):
    """
    Subtract a level expression from another.
    """

    def __call__(self, expr1, expr2, as_size):
        if as_size:
            return expr1 + expr2 - 1
        return expr1 - expr2

    def invert(self, expr1, expr2, dimensions, levels, dim_indices, lvl_indices, idx):
        rl, re = _find_range(levels, expr1, expr2)
        if re == expr2:
            dim_indices[dimensions.index(expr1)] = lvl_indices[rl] + lvl_indices[idx]
        else:
            assert re == expr1
            dim_indices[dimensions.index(expr2)] = lvl_indices[rl] - lvl_indices[idx]

    def __repr__(self):
        return "-"


class Divide(Operator):
    """
    Divide a level expression by an integer.
    """

    def __call__(self, expr1, expr2, as_size):
        if as_size:
            assert expr2 > 0, f"Internal error: {expr2}"
            if expr1 % expr2 != 0:
                raise NotImplementedError(f"block size {expr2} does not evenly divide size {expr1}")
        return expr1 // expr2

    def invert(self, expr1, expr2, dimensions, levels, dim_indices, lvl_indices, idx):
        i = dimensions.index(expr1)
        dim_indices[i] = lvl_indices[idx] * expr2

    def __repr__(self):
        return "//"


class Modulo(Operator):
    """
    The integer modulo of a level expression.
    """

    def __call__(self, expr1, expr2, as_size):
        if as_size:
            assert expr2 > 0, f"Internal error: {expr2}"
            return expr2
        return expr1 % expr2

    def invert(self, expr1, expr2, dimensions, levels, dim_indices, lvl_indices, idx):
        i = dimensions.index(expr1)
        dim_indices[i] += lvl_indices[idx]  # update (seen after div)

    def __repr__(self):
        return "%"


class LevelExpr:
    """
    A (binary) level expression is a triple: dimension object or level expression,
    operator, dimension or level expression or int.

    Args:
        expression1: a Dimension or LevelExpr object.
        operator: operator derived from Operator (Add, Subtract, Divide, Modulo).
        expression2: a Dimension or LevelExpr object, or an integer.
    """

    __slots__ = ("expression1", "operator", "expression2")

    def __init__(self, *, expression1, operator, expression2):
        lhs_level_expr_types = Dimension, LevelExpr
        rhs_level_expr_types = lhs_level_expr_types + (int,)
        if not isinstance(expression1, lhs_level_expr_types):
            raise TypeError(f"LHS expression {expression1} must have one of these types: {lhs_level_expr_types}.")
        if not isinstance(operator, Operator):
            raise TypeError(f"Operator {operator} must be derived from Operator.")
        if not isinstance(expression2, rhs_level_expr_types):
            raise TypeError(f"RHS expression {expression2} must have one of these types: {rhs_level_expr_types}.")
        self.expression1 = expression1
        self.operator = operator
        self.expression2 = expression2

    def __repr__(self):
        return f"({self.expression1} {self.operator} {self.expression2})"

    def __add__(self, expression):
        return LevelExpr(expression1=self, operator=Add(), expression2=expression)

    def __sub__(self, expression):
        return LevelExpr(expression1=self, operator=Subtract(), expression2=expression)

    def __floordiv__(self, divisor):
        return LevelExpr(expression1=self, operator=Divide(), expression2=divisor)

    def __mod__(self, modulus):
        return LevelExpr(expression1=self, operator=Modulo(), expression2=modulus)

    def evaluate(self, dimensions, dim_indices, as_size):
        e1 = self.expression1.evaluate(dimensions, dim_indices, as_size)
        e2 = (
            self.expression2.evaluate(dimensions, dim_indices, as_size)
            if not isinstance(self.expression2, int)
            else self.expression2
        )
        return self.operator(e1, e2, as_size)

    def invert(self, dimensions, levels, dim_indices, lvl_indices, idx):
        self.operator.invert(
            self.expression1,
            self.expression2,
            dimensions,
            levels,
            dim_indices,
            lvl_indices,
            idx,
        )


class Dimension:
    """
    A dimension object encapsulates a dimension name.

    Args:
        dimension_name: A name for the dimension.
    """

    __slots__ = ("dimension_name",)

    def __init__(self, *, dimension_name):
        if not isinstance(dimension_name, str):
            raise TypeError(f"The dimension name {dimension_name} must be a string.")

        self.dimension_name = dimension_name

    def __repr__(self):
        return f"{self.dimension_name}"

    def __add__(self, expression):
        return LevelExpr(expression1=self, operator=Add(), expression2=expression)

    def __sub__(self, expression):
        return LevelExpr(expression1=self, operator=Subtract(), expression2=expression)

    def __floordiv__(self, divisor):
        return LevelExpr(expression1=self, operator=Divide(), expression2=divisor)

    def __mod__(self, modulus):
        return LevelExpr(expression1=self, operator=Modulo(), expression2=modulus)

    def evaluate(self, dimensions, dim_indices, as_size):
        return dim_indices[dimensions.index(self)]

    def invert(self, dimensions, levels, dim_indices, lvl_indices, idx):
        dim_indices[dimensions.index(self)] = lvl_indices[idx]


class TensorFormat:
    """
    A universal sparse tensor format maps dimension specifications
    (dimensions for short) to level specifications (levels for short).

    Args:
        dimensions: a sequence of Dimension objects.
        levels: an ordered dictionary from a LevelExpr or Dimension object
             to a LevelFormat, or a (LevelFormat, LevelProperty) pair.
        name: a name for the format as a string. If the format corresponds to an
             existing format like CSR, COO, etc. use the canonical name. A name
             will be generated based on the level specification if none is provided.
    """

    __slots__ = ("_dimensions", "_levels", "_name", "_is_identity", "_is_ordered", "_is_unique")

    def __init__(self, dimensions, levels, *, name=None):
        self._dimensions = dimensions
        self._levels = levels

        # TODO: create a better default name.
        if name is None:
            name = f"{levels}".replace("\n", " ")
        if not isinstance(name, str):
            raise TypeError(f"The name {name} must be a string.")
        self._name = name

        # Perform semantic checks on syntactic structure.
        # Throws a TypeError on failure.
        self._semantically_validate()

        # Cached properties.
        self._is_identity = (
            all(isinstance(lvl, Dimension) and self.dimensions.index(lvl) == idx for idx, lvl in enumerate(levels))
            if self.num_dimensions == self.num_levels
            else False
        )
        self._is_ordered = all(is_ordered(lvl[1]) if isinstance(lvl, tuple) else True for lvl in levels.values())
        self._is_unique = any(is_unique(lvl[1]) if isinstance(lvl, tuple) else True for lvl in levels.values())

    @property
    def name(self):
        """
        Get the name of the sparse tensor format.
        """
        return self._name

    @property
    def num_dimensions(self):
        """
        Get the number of dimensions (rank of tensor).
        """
        return len(self._dimensions)

    @property
    def num_levels(self):
        """
        Get the number of levels (rank of storage).
        """
        return len(self._levels)

    @property
    def dimensions(self):
        """
        Get the dimension specifications for this format.
        """
        return self._dimensions

    @property
    def levels(self):
        """
        Get the level specifications for this format.
        """
        return self._levels

    @property
    def is_identity(self):
        """
        Determine whether the format uses identity mapping between dimensions and levels.
        """
        return self._is_identity

    @property
    def is_ordered(self):
        """
        Determine whether the format is ordered.
        """
        return self._is_ordered

    @property
    def is_unique(self):
        """
        Determine whether the format is unique.
        """
        # TODO: imprecise, but works for expected cases like
        # e.g. COO3 (compressed(!U), singleton(!U), singleton(U)) is still unique
        # but  COO3 (compressed(!U), singleton(!U), singleton(!U)) is nonunique
        return self._is_unique

    def _semantically_validate(self):
        """
        Semantically validates the sparse tensor format. The grammar allows for specifying
        a very wide range of sparse tensor formats. However, not every syntactically
        valid input is also semanticaly valid. This method throws a TypeError for
        tensor formats where the following constraints are not satisfied :

          (1) the input syntax has an actual implementation
          (2) the dim-to-level mapping is easily invertible

        Note that currently the checks are **stricter** than really required.
        However, developers that relax the checks below are also responsible for
        ensuring that constraints (1) and (2) are still satisfied.
        """
        # Reject incorrect dimension specifications types.
        if not (isinstance(self.dimensions, Sequence) and all(isinstance(d, Dimension) for d in self.dimensions)):
            raise TypeError(f"Dimension specifications {self.dimensions} must be a sequence of Dimension objects.")

        # Reject dimension specifications that are not unique names (e.g. (i,i)).
        all_dims = set(self.dimensions)
        if len(all_dims) != len(self.dimensions):
            raise TypeError(f"Dimension specifications {self.dimensions} has repeated identifiers.")

        # Reject incorrect level specification types.
        if not (isinstance(self.levels, dict)):
            raise TypeError(f"Level specifications {self.levels} must be an ordered dictionary.")

        # Reject incorrect level specification structure.
        used_dims = set()
        range_dims = {}
        block_dims = {}
        batch = 0
        for idx, (k, v) in enumerate(self.levels.items()):
            # Check the level expression.
            if isinstance(k, Dimension):
                if k not in self.dimensions:
                    raise TypeError(f"Dimension {k} does not appear in {self.dimensions}.")
                used_dims.add(k)
            elif isinstance(k, LevelExpr):
                # Check for unnested level expressions and impose some constraints.
                # We expect (Add, Subtract, Divide, Modulo) only.
                if isinstance(k.operator, (Add, Subtract)):
                    # Only allow for direct expressions like i - j.
                    if k.expression1 not in self.dimensions:
                        raise TypeError(f"LHS {k.expression1} in {k} does not appear in {self.dimensions}.")
                    if k.expression2 not in self.dimensions:
                        raise TypeError(f"RHS {k.expression2} in {k} does not appear in {self.dimensions}.")
                    used_dims.add(k.expression1)
                    used_dims.add(k.expression2)
                    # Find matching range.
                    if k.expression1 in range_dims or k.expression2 in range_dims:
                        raise TypeError(f"Operation {k} reuses dimension of a prior range computation.")
                    range_dims[k.expression1] = k.expression2
                    range_dims[k.expression2] = k.expression1
                elif isinstance(k.operator, (Divide, Modulo)):
                    # Only allow for direct expressions like i % 3.
                    if k.expression1 not in self.dimensions:
                        raise TypeError(f"LHS {k.expression1} in {k} does not appear in {self.dimensions}.")
                    if not isinstance(k.expression2, int):
                        raise TypeError(f"RHS {k.expression2} in {k} must be an integer.")
                    elif k.expression2 <= 0:
                        raise TypeError(f"RHS {k.expression2} in {k} must be strictly positive integer.")
                    used_dims.add(k.expression1)
                    # Find matching blocking.
                    if isinstance(k.operator, Divide):
                        if k.expression1 in block_dims:
                            raise TypeError(f"Division {k} reuses dimension of a prior division.")
                        block_dims[k.expression1] = k.expression2
                    elif isinstance(k.operator, Modulo):
                        if block_dims.get(k.expression1, 0) != k.expression2:
                            raise TypeError(f"Modulo {k} does not match any prior division.")
                        block_dims[k.expression1] = 0  # avoid further usage
                else:
                    raise TypeError(f"Unexpected operator in level expression {k}.")
            else:
                raise TypeError(f"Level expression {k} must be a Dimension or LevelExpr object.")

            # Inspect the level format for explicit or implicit properties.
            if isinstance(v, tuple):
                fmt, prop = v
            else:
                fmt, prop = v, LevelProperty.ORDERED

            # Check level format.
            if not isinstance(fmt, LevelFormat):
                raise TypeError(f"Invalid level format {fmt}.")
            elif fmt == LevelFormat.RANGE:
                if not isinstance(k, Dimension):
                    raise TypeError(f"Range uses compound level expression {k}.")
                other = range_dims.get(k, 0)
                if other == 0:
                    raise TypeError(f"Range uses dimension {k} that is not uniquely defined.")
                # Consume range tuple.
                range_dims[k] = 0
                range_dims[other] = 0

            # Verify batching.
            if fmt == LevelFormat.BATCH:
                if k != self.dimensions[idx]:
                    raise TypeError(f"Batch uses non-identity {k} for {self.dimensions[idx]}.")
                if batch == -1:
                    raise TypeError(f"Batch is used in inner level {idx}.")
                batch += 1
            elif batch > 0 and fmt != LevelFormat.DENSE:
                raise TypeError("Batch levels must end in a dense level.")
            else:
                batch = -1

            # Check level properties.
            if isinstance(prop, Sequence):
                invalid = []
                for p_ in prop:
                    if not isinstance(p_, LevelProperty):
                        raise TypeError(f"Invalid level property {p_}.")
                    if LevelPropertyInverse[p_] in prop:
                        invalid += (p_,)
                if invalid:
                    raise TypeError(f"Invalid level property combination: {invalid}.")
            elif not isinstance(prop, LevelProperty) and not isinstance(prop, int):
                raise TypeError(f"Invalid level property {prop}.")

        # Check that every dimension is used eventually.
        if len(all_dims) != len(used_dims):
            raise TypeError(f"The following dimensions are not used: {all_dims - used_dims}.")

        # Check that every div is matched by mod.
        unmatched = [key for key, value in block_dims.items() if value != 0]
        if len(unmatched) != 0:
            raise TypeError(f"Some division dimensions are not matched by modulo: {unmatched}.")

        # Check that every add/sub is matched by range.
        unmatched = [key for key, value in range_dims.items() if value != 0]
        if len(unmatched) != 0:
            raise TypeError(f"Some add/sub dimensions are not matched by range: {unmatched}.")

        # Check that batching ends in dense.
        if batch > 0:
            raise TypeError("Batch levels are not properly closed by a dense level.")

    def dim2lvl(self, dim_indices, as_size=False):
        """
        Maps the dimension index list to the level index list.

        Examples:
          CSR:           [0,4] -> [0,4]
          CSC:           [0,4] -> [4,0]
          BSRRight(2,2): [1,2] -> [0,1,0]
        """
        return [ls.evaluate(self.dimensions, dim_indices, as_size) for ls in self.levels]

    def lvl2dim(self, lvl_indices):
        """
        Maps the level index list to the dimension index list (the inverse of dim2lvl).
        We can safely assume that the mapping is invertible given that the tensor format
        passed validation.

        Examples:
          CSR:           [0,4]   -> [0,4]
          CSC:           [4,0]   -> [0,4]
          BSRRight(2,2): [0,1,0] -> [1,2]
        """
        dim_indices = [0] * self.num_dimensions  # pre-populate
        for idx, ls in enumerate(self.levels):
            ls.invert(self.dimensions, self.levels, dim_indices, lvl_indices, idx)
        return dim_indices

    def __repr__(self):
        return (
            f"{self.dimensions} -> ("
            + ", ".join([f"{k}: {v if isinstance(v, Sequence) else repr(v)}" for k, v in self.levels.items()])
            + ")"
        )


class NamedFormats:
    """
    A number of pre-defined common tensor formats (direct or using builders).
    Clients can always construct these or other tensor formats directly.
    """

    i, j, k, b = (
        Dimension(dimension_name="i"),
        Dimension(dimension_name="j"),
        Dimension(dimension_name="k"),
        Dimension(dimension_name="batch"),
    )

    # Scalar format.
    Scalar = TensorFormat([], {}, name="Scalar")

    # Vector formats.
    DenseVector = TensorFormat([i], {i: LevelFormat.DENSE}, name="DenseVector")
    SparseVector = TensorFormat([i], {i: LevelFormat.COMPRESSED}, name="SparseVector")

    # Dense matrix formats.
    DenseMatrixRight = TensorFormat(
        [i, j],
        {i: LevelFormat.DENSE, j: LevelFormat.DENSE},
        name="DenseMatrixRight",
    )
    DenseMatrixLeft = TensorFormat(
        [i, j],
        {j: LevelFormat.DENSE, i: LevelFormat.DENSE},
        name="DenseMatrixLeft",
    )

    # Sparse matrix formats.
    COO = TensorFormat(
        [i, j],
        {
            i: (LevelFormat.COMPRESSED, LevelProperty.NONUNIQUE),
            j: LevelFormat.SINGLETON,
        },
        name="COO",
    )
    CSR = TensorFormat([i, j], {i: LevelFormat.DENSE, j: LevelFormat.COMPRESSED}, name="CSR")
    CSC = TensorFormat([i, j], {j: LevelFormat.DENSE, i: LevelFormat.COMPRESSED}, name="CSC")
    DCSR = TensorFormat(
        [i, j],
        {i: LevelFormat.COMPRESSED, j: LevelFormat.COMPRESSED},
        name="DCSR",
    )
    DCSC = TensorFormat(
        [i, j],
        {j: LevelFormat.COMPRESSED, i: LevelFormat.COMPRESSED},
        name="DCSC",
    )
    CROW = TensorFormat([i, j], {i: LevelFormat.COMPRESSED, j: LevelFormat.DENSE}, name="CROW")
    CCOL = TensorFormat([i, j], {j: LevelFormat.COMPRESSED, i: LevelFormat.DENSE}, name="CCOL")

    # Diagonal matrix formats.
    DIAI = TensorFormat(
        [i, j],
        {j - i: LevelFormat.COMPRESSED, i: LevelFormat.RANGE},
        name="DIAI",
    )
    DIAJ = TensorFormat(
        [i, j],
        {j - i: LevelFormat.COMPRESSED, j: LevelFormat.RANGE},
        name="DIAJ",
    )
    SkewDIAI = TensorFormat(
        [i, j],
        {i + j: LevelFormat.COMPRESSED, i: LevelFormat.RANGE},
        name="SkewDIAI",
    )
    SkewDIAJ = TensorFormat(
        [i, j],
        {i + j: LevelFormat.COMPRESSED, j: LevelFormat.RANGE},
        name="SkewDIAJ",
    )

    # Batched sparse matrices.
    BatchedCSR = TensorFormat(
        [b, i, j],
        {b: LevelFormat.DENSE, i: LevelFormat.DENSE, j: LevelFormat.COMPRESSED},
        name="BatchedCSR",
    )
    BatchedDIAINonUniform = TensorFormat(
        [b, i, j],
        {b: LevelFormat.DENSE, j - i: LevelFormat.COMPRESSED, i: LevelFormat.RANGE},
        name="BatchedDIAINonUniform",
    )
    BatchedDIAIUniform = TensorFormat(
        [b, i, j],
        {j - i: LevelFormat.COMPRESSED, b: LevelFormat.DENSE, i: LevelFormat.RANGE},
        name="BatchedDIAIUniform",
    )

    #
    # Tensor format builders for vectors, matrices, and tensors.
    #

    @staticmethod
    def BlockVector(blocksize):
        """
        Block vector format.

        Args:
            blocksize: an integer specifying the blocksize in the i'th dimension.
        """
        i = NamedFormats.i
        name = f"BlockVector{blocksize}"
        return TensorFormat(
            [i],
            {i // blocksize: LevelFormat.COMPRESSED, i % blocksize: LevelFormat.DENSE},
            name=name,
        )

    @staticmethod
    def BSRRight(blocksize):
        """
        BSRRight matrix format.

        Args:
            blocksize: a pair specifying the blocksize in the i'th and j'th dimensions.
        """
        i, j = NamedFormats.i, NamedFormats.j
        m, n = blocksize
        name = f"BSRRight{m}x{n}"
        return TensorFormat(
            [i, j],
            {
                i // m: LevelFormat.DENSE,
                j // n: LevelFormat.COMPRESSED,
                i % m: LevelFormat.DENSE,
                j % n: LevelFormat.DENSE,
            },
            name=name,
        )

    @staticmethod
    def BSRLeft(blocksize):
        """
        BSRLeft matrix format.

        Args:
            blocksize: a pair specifying the blocksize in the i'th and j'th dimensions.
        """
        i, j = NamedFormats.i, NamedFormats.j
        m, n = blocksize
        name = f"BSRLeft{m}x{n}"
        return TensorFormat(
            [i, j],
            {
                i // m: LevelFormat.DENSE,
                j // n: LevelFormat.COMPRESSED,
                j % n: LevelFormat.DENSE,
                i % m: LevelFormat.DENSE,
            },
            name=name,
        )

    @staticmethod
    def BSCRight(blocksize):
        """
        BSCRight matrix format.

        Args:
            blocksize: a pair specifying the blocksize in the i'th and j'th dimensions.
        """
        i, j = NamedFormats.i, NamedFormats.j
        m, n = blocksize
        name = f"BSCRow{m}x{n}"
        return TensorFormat(
            [i, j],
            {
                j // n: LevelFormat.DENSE,
                i // m: LevelFormat.COMPRESSED,
                i % m: LevelFormat.DENSE,
                j % n: LevelFormat.DENSE,
            },
            name=name,
        )

    @staticmethod
    def BSCLeft(blocksize):
        """
        BSCLeft matrix format.

        Args:
            blocksize: a pair specifying the blocksize in the i'th and j'th dimensions.
        """
        i, j = NamedFormats.i, NamedFormats.j
        m, n = blocksize
        name = f"BSCCol{m}x{n}"
        return TensorFormat(
            [i, j],
            {
                j // n: LevelFormat.DENSE,
                i // m: LevelFormat.COMPRESSED,
                j % n: LevelFormat.DENSE,
                i % m: LevelFormat.DENSE,
            },
            name=name,
        )

    @staticmethod
    def DELTA(delta):
        """
        Delta compression matrix format.

        Args:
            delta: number of bits for delta
        """
        i, j = NamedFormats.i, NamedFormats.j
        name = f"Delta{delta}"
        return TensorFormat(
            [i, j],
            {i: LevelFormat.DENSE, j: (LevelFormat.DELTA, delta)},
            name=name,
        )

    @staticmethod
    def BSR3(blocksize):
        """
        3D BSR format.

        Args:
            blocksize: a triple specifying the blocksize in the i'th, j'th, and k'th
                dimensions.
        """
        i, j, k = NamedFormats.i, NamedFormats.j, NamedFormats.k
        m, n, o = blocksize
        name = f"BSR3{m}x{n}x{o}"
        return TensorFormat(
            [i, j, k],
            {
                i // m: LevelFormat.DENSE,
                j // n: LevelFormat.COMPRESSED,
                k // o: LevelFormat.COMPRESSED,
                i % m: LevelFormat.DENSE,
                j % n: LevelFormat.DENSE,
                k % o: LevelFormat.DENSE,
            },
            name=name,
        )

    @staticmethod
    def DensedRight(dim):
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]
        return TensorFormat(
            dims,
            {dims[d]: LevelFormat.DENSE for d in range(dim)},
            name=f"Dense{dim}Right",
        )

    @staticmethod
    def DensedLeft(dim):
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]
        return TensorFormat(
            dims,
            {dims[dim - d - 1]: LevelFormat.DENSE for d in range(dim)},
            name=f"Dense{dim}Left",
        )

    @staticmethod
    def COOd(sparse_dim, dense_dim=0):
        dim = sparse_dim + dense_dim
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]
        return TensorFormat(
            dims,
            {
                dims[d]: (
                    (LevelFormat.COMPRESSED if sparse_dim == 1 else (LevelFormat.COMPRESSED, LevelProperty.NONUNIQUE))
                    if d == 0
                    else LevelFormat.SINGLETON
                    if d < sparse_dim
                    else LevelFormat.DENSE
                )
                for d in range(dim)
            },
            name=("COO" if dim == 2 else f"COO{dim}") if dense_dim == 0 else None,
        )

    @staticmethod
    def CSRd(batch_dim=0, dense_dim=0):
        dim = batch_dim + 2 + dense_dim
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]

        name = "CSR"
        if batch_dim > 0:
            name += f"-b{batch_dim}"
        if dense_dim > 0:
            name += f"-d{dense_dim}"

        return TensorFormat(
            dims,
            {
                dims[d]:
                # Format
                LevelFormat.BATCH if d < batch_dim else LevelFormat.COMPRESSED if d == batch_dim + 1 else LevelFormat.DENSE
                for d in range(dim)
            },
            name=name,
        )

    @staticmethod
    def CSCd(batch_dim=0, dense_dim=0):
        dim = batch_dim + 2 + dense_dim
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]

        name = "CSC"
        if batch_dim > 0:
            name += f"-b{batch_dim}"
        if dense_dim > 0:
            name += f"-d{dense_dim}"

        return TensorFormat(
            dims,
            {
                dims[d + 1 if d == batch_dim else d - 1 if d == batch_dim + 1 else d]:
                # Format
                LevelFormat.BATCH if d < batch_dim else LevelFormat.COMPRESSED if d == batch_dim + 1 else LevelFormat.DENSE
                for d in range(dim)
            },
            name=name,
        )

    @staticmethod
    def BSRRightd(blocksize, batch_dim=0, dense_dim=0):
        m, n = blocksize
        dim = batch_dim + 2 + dense_dim
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]

        name = f"BSRRight{m}x{n}"
        if batch_dim > 0:
            name += f"-b{batch_dim}"
        if dense_dim > 0:
            name += f"-d{dense_dim}"

        return TensorFormat(
            dims,
            {dims[d]: LevelFormat.BATCH for d in range(batch_dim)}
            | {
                dims[batch_dim] // m: LevelFormat.DENSE,
                dims[batch_dim + 1] // n: LevelFormat.COMPRESSED,
                dims[batch_dim] % m: LevelFormat.DENSE,
                dims[batch_dim + 1] % n: LevelFormat.DENSE,
            }
            | {dims[d]: LevelFormat.DENSE for d in range(batch_dim + 2, dim)},
            name=name,
        )

    @staticmethod
    def BSCRightd(blocksize, batch_dim=0, dense_dim=0):
        m, n = blocksize
        dim = batch_dim + 2 + dense_dim
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]

        name = f"BSCRight{m}x{n}"
        if batch_dim > 0:
            name += f"-b{batch_dim}"
        if dense_dim > 0:
            name += f"-d{dense_dim}"

        return TensorFormat(
            dims,
            {dims[d]: LevelFormat.BATCH for d in range(batch_dim)}
            | {
                dims[batch_dim + 1] // n: LevelFormat.DENSE,
                dims[batch_dim] // m: LevelFormat.COMPRESSED,
                dims[batch_dim] % m: LevelFormat.DENSE,
                dims[batch_dim + 1] % n: LevelFormat.DENSE,
            }
            | {dims[d]: LevelFormat.DENSE for d in range(batch_dim + 2, dim)},
            name=name,
        )

    @staticmethod
    def CSFd(dim):
        dims = [Dimension(dimension_name=f"{chr(ord('i') + d)}") for d in range(dim)]
        return TensorFormat(
            dims,
            {dims[d]: LevelFormat.COMPRESSED for d in range(dim)},
            name=f"CSF{dim}",
        )
