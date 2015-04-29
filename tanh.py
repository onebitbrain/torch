# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:43:02 2015

@author: Sam
"""

import numpy as np
from module import Module


class tanh(Module):
    """
    """
    def update_output(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def update_grad_inputs(self, inputs, grad_output):
        tanh = self.output
        self.grad_inputs = self.grad_inputs * (1 - tanh * tanh)
        return self.grad_inputs
