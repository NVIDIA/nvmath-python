# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import PrimitiveModel
from numba.cuda.extending import models, register_model, types


class LLVMArray(types.Array):
    def __init__(self, dtype, size):
        super().__init__(dtype, layout="C", ndim=1)
        self.size = size

    def __str__(self):
        return f"LLVMArray(dtype={self.dtype}, size={self.size})"

    def __repr__(self):
        return f"LLVMArray(dtype={self.dtype}, size={self.size})"


@register_model(LLVMArray)
class LLVMArrayModel(models.PrimitiveModel):
    def __init__(self, dmm: DataModelManager, fe_type: LLVMArray):
        dtype_model: PrimitiveModel = dmm.lookup(fe_type.dtype)
        llvm_type = dtype_model.be_type
        be_type = ir.ArrayType(llvm_type, fe_type.size)
        super().__init__(dmm, fe_type, be_type)
