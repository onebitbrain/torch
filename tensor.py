"""
Torch7 re-implemented in python.
"""

import numpy as np


__all__ = ['FloatTensor', 'DoubleTensor', 'CharTensor', 'ByteTensor',
           'ShortTensor', 'IntTensor', 'LongTensor', 'Tensor']


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


def empty_like(tensor, dtype=None, order='K'):
    return np.empty_like(tensor, dtype, order, subok=True)


def ones_like(tensor, dtype=None, order='K'):
    return np.ones_like(tensor, dtype, order, subok=True)


def zeros_like(tensor, dtype=None, order='K'):
    return np.zeros_like(tensor, dtype, order, subok=True)


def Tensor(arg, tensor_type=FloatTensor):
    if isinstance(arg, np.ndarray):
        return arg.view(tensor_type)
    return 
