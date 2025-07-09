# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
An interface class to query algorithm capabilities and configure it.
"""

__all__ = ["Algorithm"]

import dataclasses


from nvmath.linalg.advanced._configuration import AlgorithmCapabilities
from nvmath.linalg._internal.algo_cap_ifc import AlgoCapInterface
from nvmath.linalg._internal.algo_config_ifc import AlgoConfigInterface
from nvmath.bindings.cublasLt import MatmulHeuristicResult  # type: ignore


class Algorithm:
    """
    An interface class to query algorithm capabilities and configure the algorithm.

    Note that objects of this type should not be constructed directly by the user.
    """

    def __init__(self, algorithm):
        assert isinstance(algorithm, MatmulHeuristicResult), "Internal error."
        self.algorithm = algorithm
        self.cap_ifc = AlgoCapInterface(algorithm)
        self.config_ifc = AlgoConfigInterface(algorithm)

    @property
    def capabilities(self):
        """
        Return the capabilities of this algorithm as a
        :class:`nvmath.linalg.advanced.AlgorithmCapabilities` dataclass.
        """
        names = [field.name for field in dataclasses.fields(AlgorithmCapabilities)]
        _capabilities = {}
        for name in names:
            # Not all capabilities in the enum are supported in all CTK versions.
            try:
                _capabilities[name] = getattr(self.cap_ifc, name)
            except:
                pass
        return AlgorithmCapabilities(**_capabilities)

    @property
    def algorithm_id(self):
        "The ID of the algorithm (integer)."
        return self.config_ifc.id

    @property
    def tile(self):
        """A tuple representing the tile (see MatmulAlgoConfigAttribute.TILE_ID).
        The value provided must be one of the `tile_ids` in the algorithm capabilities."""
        return self.config_ifc.tile_id

    @tile.setter
    def tile(self, tile):
        self.config_ifc.tile_id = tile

    @property
    def stages(self):
        """A tuple representing the stages (see MatmulAlgoConfigAttribute.STAGES_ID).
        The value provided must be one of the `stages_ids` in the algorithm capabilities."""
        return self.config_ifc.stages_id

    @stages.setter
    def stages(self, stages):
        self.config_ifc.stages_id = stages

    @property
    def split_k(self):
        """The number of split-k steps (see MatmulAlgoConfigAttribute.SPLITK_NUM).

        This can be set only if `splitk_support` is 1 in the algorithm capabilities."""
        return self.config_ifc.splitk_num

    @split_k.setter
    def split_k(self, number):
        self.config_ifc.splitk_num = number

    @property
    def reduction_scheme(self):
        """The reduction scheme used (see MatmulAlgoConfigAttribute.REDUCTION_SCHEME).

        The value provided must be consistent with the `reduction_scheme_mask` in the
        algorithm capabilities."""
        return self.config_ifc.reduction_scheme

    @reduction_scheme.setter
    def reduction_scheme(self, scheme_id):
        self.config_ifc.reduction_scheme = scheme_id

    @property
    def cta_swizzling(self):
        """A flag indicating CTA swizzling (see MatmulAlgoConfigAttribute.CTA_SWIZZLING).

        This can be set only if `cta_swizzling` is 1 in the algorithm capabilities."""
        return self.config_ifc.cta_swizzling

    @cta_swizzling.setter
    def cta_swizzling(self, flag: bool):
        self.config_ifc.cta_swizzling = flag

    @property
    def custom_option(self):
        """A value indicating the custom option (see
        MatmulAlgoConfigAttribute.CUSTOM_OPTION).

        The value provided must be less than `custom_option_max` in the algorithm
        capabilities."""
        return self.config_ifc.custom_option

    @custom_option.setter
    def custom_option(self, value: int):
        self.config_ifc.custom_option = value

    @property
    def inner_shape(self):
        """A value indicating the inner shape (see
        MatmulAlgoConfigAttribute.INNER_SHAPE_ID)."""
        return self.config_ifc.inner_shape_id

    @inner_shape.setter
    def inner_shape(self, shape):
        self.config_ifc.inner_shape_id = shape

    @property
    def cluster_shape(self):
        """A tuple representing the cluster shape (see
        MatmulAlgoConfigAttribute.CLUSTER_SHAPE_ID).

        The value provided must be one of the `cluster_shape_ids` in the algorithm
        capabilities."""
        return self.config_ifc.cluster_shape_id

    @cluster_shape.setter
    def cluster_shape(self, shape):
        self.config_ifc.cluster_shape_id = shape
