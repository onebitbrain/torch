# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:08:13 2015

@author: Sam
"""


from networkx import DiGraph
import networkx.topological_sort as ts
import networkx.topological_sort_recursive as tsr
import networkx.draw as d
import networkx.draw_graphviz as dg


__all__ = ['Graph']


class Graph(DiGraph):

    def topological_sort(self, nbunch=None, reverse=False):
        return ts(self, nbunch, reverse)

    def topological_sort_recursive(self, nbunch=None, reverse=False):
        return tsr(self, nbunch, reverse)

    def draw(self, pos=None, ax=None, **kwds):
        return d(self, pos, ax, **kwds)

    def draw_graphviz(self, prog='neato', **kwargs):
        return dg(self, prog, **kwargs)
