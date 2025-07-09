# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities to convert tuple like enumerators to tuples, along with forward and inverse maps.
"""

__all__ = [
    "CLUSTER_SHAPES",
    "CLUSTER_SHAPE_TO_ENUM",
    "ENUM_TO_CLUSTER_SHAPE",
    "MATMUL_STAGES",
    "MATMUL_STAGE_TO_ENUM",
    "ENUM_TO_MATMUL_STAGE",
    "MATMUL_TILES",
    "MATMUL_TILE_TO_ENUM",
    "ENUM_TO_MATMUL_TILE",
]

import re

from nvmath.bindings import cublasLt as cublaslt


def integer_or_string(value):
    try:
        value = int(value)
    except ValueError:
        ...
    return value


def create_valid_tuples_from_enum(enum, prefix, *, expr=r"(?:(\d+)x(\d+|\w+)(?:x(\d+))?|(AUTO|UNDEFINED))"):
    """
    Create a sequence of tuples representing the allowed combinations for the given
    enumeration.
    """

    combinations = []
    enumerator_to_value = {}
    value_to_enumerator = {}
    expr = prefix + expr
    for e in enum:
        m = re.match(expr, e.name)
        if m:
            # print(m.groups())
            if m.group(4):
                v = m.group(4)
            else:
                groups = m.groups()[: m.groups().index(None)]
                v = tuple(map(integer_or_string, (g for g in groups)))
            combinations.append(v)
            value_to_enumerator[v] = e
            enumerator_to_value[e] = v

    return tuple(combinations), value_to_enumerator, enumerator_to_value


CLUSTER_SHAPES, CLUSTER_SHAPE_TO_ENUM, ENUM_TO_CLUSTER_SHAPE = create_valid_tuples_from_enum(cublaslt.ClusterShape, "SHAPE_")

MATMUL_STAGES, MATMUL_STAGE_TO_ENUM, ENUM_TO_MATMUL_STAGE = create_valid_tuples_from_enum(cublaslt.MatmulStages, "STAGES_")

MATMUL_TILES, MATMUL_TILE_TO_ENUM, ENUM_TO_MATMUL_TILE = create_valid_tuples_from_enum(cublaslt.MatmulTile, "TILE_")
