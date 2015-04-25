# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:33:23 2015

@author: Sam
"""


from module import *
from graph import *


def get_total_grad_output(node):
    grad_output = node[grad_output]
    assert is_tensor(grad_output), 'expecting gradients to sum'
    return grad_output.sum()  # not sure


class GraphModule(Module):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.bg = Graph()
        self.bg.add_nodes_from(outputs)
        self.fg = self.bg.reverse()
