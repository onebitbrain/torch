# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:38:50 2015

@author: Sam
"""
from module import Module
import numpy as np


class Dropout(Module):

    def __init__(self, probability=0.5, scaled=False):
        assert probability < 1 and probability > 0, 'probability must be (0,1)'
        Module.__init__(self)
        self.prob = probability
        self.train = True
        self.scaled = scaled
        self.noise = torch.Tensor()

    def update_output(self, inputs):
        if self.train:
            self.noise = np.random.binomial(1, self.prob, inputs.size())
            self.output = self.noise * inputs
        else:
            self.output = inputs * (1 - self.prob)
        return self.output

    def update_grad_inputs(self, inputs, grad_output):
        if self.train:
            self.grad_inputs = self.noise * grad_output
        else:
            raise 'backprop only defined while training'
        return self.grad_inputs

    def set_probability(self, probability):
        self.prob = probability
