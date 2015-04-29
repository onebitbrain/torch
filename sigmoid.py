# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:33:18 2015

@author: Sam
"""

import numpy as np
from module import Module


class sigmoid(Module):
    """
    """
    def update_output(self, inputs):
        return 1. / (1 + np.exp(-inputs))

    def update_grad_inputs(self, inputs, grad_output):
        sig = self.update_output(inputs)
        return sig * (1 - sig)
