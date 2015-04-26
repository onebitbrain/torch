# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:24:51 2015

@author: Sam
"""
import numpy as np
from module import Module
from tensor import Tensor


class Linear(Module):
    """
    """
    def __init__(self, input_size, output_size):
        Module.__init__(self)
        self.weight = Tensor(output_size, input_size)
        self.bias = Tensor(output_size)
        self.grad_weight = Tensor(output_size, input_size)
        self.grad_bias = Tensor(output_size)
        self.reset()

    def reset(stdv):
        if stdv:
            stdv = stdv * np.sqrt(3)
        else:
            stdv = 1./np.sqrt(self.weight.shape[1])
        if old_seed:  # fix
            pass
        else:
            self.weight = Tensor(np.random.uniform(-stdv, stdv, self.weight.shape))
            self.bias = Tensor(np.random.uniform(-stdv, stdv, self.bias.shape))
        return self  #?

    def update_output(self, inputs):
        pass

    def update_grad_inputs(self, inputs, grad_output):
        pass

    def acc_grad_params(self, inputs, grad_output, scale):
        pass

    def __str__(self):
        return str('(%d -> %d)', self.weight.shape[1], self.bias.shape[0])
