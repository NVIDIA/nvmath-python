# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use tensors (or ndarray-like objects) from different libraries.
"""

from abc import ABC, abstractmethod

from . import typemaps

class Tensor(ABC):
    """
    A simple wrapper type for tensors to make the API package-agnostic.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    @property
    @abstractmethod
    def data_ptr(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls, shape, **context):
        raise NotImplementedError

    @abstractmethod
    def numpy(self, stream_holder):
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def strides(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls, shape, **context):
        raise NotImplementedError

    @abstractmethod
    def to(self, device='cpu', stream_holder=None):
        raise NotImplementedError

    @abstractmethod
    def copy_(self, src, stream_holder=None):
        raise NotImplementedError

    @staticmethod
    def create_name_dtype_map(conversion_function, exception_type):
        """
        Create a map between CUDA data type names and the corresponding package dtypes for supported data types.
        """
        names = typemaps.NAME_TO_DATA_TYPE.keys()
        name_to_dtype = dict()
        for name in names:
            try:
                name_to_dtype[name] = conversion_function(name)
            except exception_type:
                pass
        return name_to_dtype

