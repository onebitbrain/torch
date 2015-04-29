# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:39:34 2015

@author: Sam
"""

import numpy as np
from module import Module


class softmax(Module):
    """
    """
    def update_output(self, inputs):
        mx = np.max(inputs)
        ex = np.exp(inputs - mx)
        return ex / np.sum(1 + ex)

    def update_grad_inputs(self, inputs, grad_output):
        sm = self.update_output(inputs)
        return sm * (1 - sm)
