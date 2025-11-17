# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import opt_einsum as oe

from .common_axes import Framework, DType, MemBackend
from .input_fixtures import get_random_input_data


class ContractionTestCase:
    def __init__(self, equation: str, shapes: Sequence[Sequence[int]]):
        # normalize the equation using opt_einsum to handle the ellipses
        if "..." in equation:
            info = oe.contract_path(equation, *shapes, shapes=True)[1]
            equation = info.eq
        else:
            equation = equation
        self._equation = equation
        self._shapes = shapes

    @property
    def equation(self):
        return self._equation

    @property
    def shapes(self):
        return self._shapes

    @property
    def num_inputs(self):
        return len(self.shapes)

    def gen_input_operands(self, framework: Framework, dtype: DType, mem_backend: MemBackend, seed: int):
        operands = []
        for i, shape in enumerate(self.shapes):
            operands.append(get_random_input_data(framework, shape, dtype, mem_backend, seed + i))
        return operands

    def _get_output_shape(self):
        if "->" in self.equation:
            output_str = self.equation.split("->")[1]
        else:
            output_str = oe.parser.find_output_str(self.equation)
        inputs = self.equation.split("->")[0].split(",")
        output_shape = oe.parser.find_output_shape(inputs, self.shapes, output_str)
        return output_shape

    def gen_random_output(self, framework: Framework, dtype: DType, mem_backend: MemBackend, seed: int):
        output_shape = self._get_output_shape()
        return get_random_input_data(framework, output_shape, dtype, mem_backend, seed)


contraction_test_cases = (
    # binary tensor contraction
    ContractionTestCase(equation="ij,jk->ik", shapes=[(2, 3), (3, 4)]),
    ContractionTestCase(equation="a,a->", shapes=[(4,), (4,)]),
    ContractionTestCase(equation="ax,a->ax", shapes=[(3, 5), (3,)]),
    ContractionTestCase(equation="ac,bd->bcda", shapes=[(2, 3), (4, 1)]),
    ContractionTestCase(equation="...,...->...", shapes=[(2, 3, 4), (2, 3, 4)]),
    # ternary tensor contraction
    ContractionTestCase(equation="ijkl,klmn,mnp->ijp", shapes=[(2, 2, 4, 5), (4, 5, 2, 3), (2, 3, 3)]),
    ContractionTestCase(equation="...,...,...->...", shapes=[(2, 4), (2, 4), (2, 4)]),
    ContractionTestCase(equation="...,...,ab->a", shapes=[(2, 4), (2, 4), (3, 5)]),
    ContractionTestCase(equation="abc,bc,x->", shapes=[(2, 3, 4), (3, 4), (5,)]),
    ContractionTestCase(equation="a,b,cd->abcd", shapes=[(2,), (3,), (5, 4)]),
)
