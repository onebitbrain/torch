# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:11 2015

@author: Sam
"""

import numpy as np
from module import Module


class SpatialConvolution(Module):
    """
    """
    def __init__ = (self, input_plane, output_plane, kw, kh, dw=1, dh=1, padding=0):
        Module.__init__(self)
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.dw = dw
        self.dh = dh
        self.kw = kw
        self.kh = kh
        self.padding = padding
        self.weight = Tensor(output_plane, input_plane, kh, kw)
        self.bias = Tensor(output_plane)
        self.grad_weight = Tensor(output_plane, input_plane, kh, kw)
        self.grad_bias = Tensor(output_plane)
        self.reset()

    def reset(stdv):
        if stdv:
            stdv = stdv * np.sqrt(3)
        else:
            stdv = 1./np.sqrt(self.kw * self.kh * self.input_plane)
        if old_seed:  # fix
            pass
        else:
            self.weight = Tensor(np.random.uniform(-stdv, stdv, self.weight.shape))
            self.bias = Tensor(np.random.uniform(-stdv, stdv, self.bias.shape))
        return self  #?

    
        