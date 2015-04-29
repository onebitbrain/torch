# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:45:25 2015

@author: Sam
"""

import numpy as np
from module import Module


class ReLU(Module):
    """
    """
    def update_output(self, inputs):
        return inputs * (inputs > 0)

    def update_grad_inputs(self, inputs, grad_output):
        return (inputs > 0).astype(np.int8)
