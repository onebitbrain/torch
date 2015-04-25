"""
Torch7 re-implemented in python.
"""

import numpy as np


__all__ = ['FloatTensor', 'DoubleTensor', 'CharTensor', 'ByteTensor',
           'ShortTensor', 'IntTensor', 'LongTensor', 'Tensor']


#class BaseTensor(np.ndarray):
#    def __new__(cls, *args, **kwargs):
#        return np.ndarray.__new__(cls, *args, **kwargs)
#
##    def __array_wrap__(self, out_arr, context=None):
##        return np.ndarray().__array_wrap__(self, out_arr, context)
#
#    def narrow(self, dim, start, stride):
#        pass


class FloatTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.float32)
        return np.ndarray.__new__(cls, *args, **kwargs)


class DoubleTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.float64)
        return np.ndarray.__new__(cls, *args, **kwargs)


class CharTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.int8)
        return np.ndarray.__new__(cls, *args, **kwargs)


class ByteTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.uint8)
        return np.ndarray.__new__(cls, *args, **kwargs)


class ShortTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.int16)
        return np.ndarray.__new__(cls, *args, **kwargs)


class IntTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.int32)
        return np.ndarray.__new__(cls, *args, **kwargs)


class LongTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs.update(dtype=np.int64)
        return np.ndarray.__new__(cls, *args, **kwargs)


def Tensor(shape, default_tensor=FloatTensor):
    return default_tensor(shape)