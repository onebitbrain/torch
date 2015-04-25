# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:24:01 2015

@author: Sam
"""
from copy import deepcopy
import numpy as np
import numpy.linalg as LA

__all__ = ['AbsCriterion', 'BCECriterion', 'MSECriterion']


class Criterion(object):
    """This is an abstract class which declares methods defined in all
    criterions. This class is serializable.  <-# need to check if this is True
    
    State variable : output
    Contains the result of the last forward(input, target) call.
    
    State variable : grad_inputs
    Contains the result of the last backward(input, target) call.
    """
    def __init__(self):
        self.output = 0

    def update_output(self, inputs, target):
        pass

    def forward(self, inputs, target):
        """Given an input and a target, compute the loss function
        associated to the criterion and return the result. In general
        input and target are tensors, but some specific criterions might
        require some other type of object.

        The output returned should be a scalar in general.

        The state variable self.output should be updated after a call to
        forward().
        """
        return self.update_output(inputs, target)

    def backward(self, inputs, target):
        """Given an input and a target, compute the gradients of the
        loss function associated to the criterion and return the result.
        In general input, target and gradInput are tensors, but some
        specific criterions might require some other type of object.

        The state variable self.grad_inputs should be updated after a
        call to backward().
        """
        return self.update_grad_inputs(inputs, target)

    def update_grad_inputs(inputs, target):
        pass

    def clone(self):
        return deepcopy(self)

    def __call__(self, inputs, target):
        self.output = self.forward(inputs, target)
        self.grad_inputs = self.backward(inputs, target)
        return self.output, self.grad_inputs


class AbsCriterion(Criterion):
    """Absolute Value

    f(y, x) = sum( |x - y| ) / n
    
    >>> abs = AbsCriterion()
    Creates a criterion that measures the mean absolute value of the
    element-wise difference between input x and target y.
    
    If x and y are d-dimensional Tensors with a total of n elements, the
    sum operation still operates over all the elements, and divides by n.

    The division by n can be avoided if one sets the internal variable
    size_average to False.
    
    >>> abs.size_average = False
    >>> target = np.array([.4, .1, .25, .25])
    >>> in1 = np.array([.25, .25, .25, .25])
    >>> in2 = np.array([.4, .1, .1, .4])
    
    >>> out1 = abs.forward(in1, target)
    >>> out2 = abs.forward(in2, target)
    >>> out1 < out2
    True
    """
    def __init__(self):
        Criterion.__init__(self)
        self.size_average = True

    def update_output(self, inputs, target):
        """Returns the same result as forward."""
        return np.sum(np.abs(inputs - target))

    def update_grad_inputs(self, inputs, target):
        """Returns the same result as backward"""
        return np.sign(inputs - target)


class BCECriterion(Criterion):
    """Binary Cross Entropy

    f(y, x) = -y * log(x) - (1 - y) * log(1 - x)

    http://en.wikipedia.org/wiki/Cross_entropy
    
    >>> target = np.array([.4, .1, .25, .25])
    >>> in1 = np.array([.25, .25, .25, .25])
    >>> in2 = np.array([.4, .1, .1, .4])
    >>> bce = BCECriterion()
    >>> out1 = bce.forward(in1, target)
    >>> out2 = bce.forward(in2, target)
    >>> out1 < out2
    True

    This is useful for measuring the error of a reconstruction in an
    auto-encoder.
    """
    def __init__(self):
        Criterion.__init__(self)
        self.size_average = True

    def update_output(self, inputs, target):
        """return -- log(input) * target + log(1 - input) * (1 - target)
            
        >>> target = np.array([.3, .2, .1, .2, .2])
        >>> in1 = np.array([.2, .2, .2, .2, .2])
        >>> in2 = np.array([.3, .1, .1, .1, .4])
        >>> bce = BCECriterion()
        >>> out1 = bce.update_output(in1, target)
        >>> out2 = bce.update_output(in2, target)
        >>> out1 < out2
        True

        """
       # term1 = np.nan_to_num(np.log(inputs)) * target
        #term2 = np.nan_to_num(np.log1p(-inputs)) * (1 - target)
        #self.output = -term1 - term2
        if self.size_average:
            self.output = target * np.log(inputs) / len(inputs)
        else:
            self.output = target * np.log(inputs)
        return self.output

    def update_grad_inputs(self, inputs, target):
        """return -- target / input - (1 - target) / (1 - input)"""
        self.grad_inputs = target/inputs - (1 - target)/(1 - inputs)
        return self.grad_inputs


class MSECriterion(Criterion):
    """mean squared error"""
    def __init__(self):
        Criterion.__init__(self)
        self.size_average = True

    def update_output(self, inputs, target):
        err = inputs - target
        z = err * err
        return z.sum()

    def update_grad_inputs(self, inputs, target):
        return LA.norm(inputs - target)


def criterion_jacobi_test_1D(criterion, inputs, target):
    """
    >>> inputs = np.random.rand(10)
    >>> target = np.random.rand(10)
    >>> cri = BCECriterion()
    >>> criterion_jacobi_test_1D(cri, inputs, target)
    True
    """
    eps = 1e-6
    _ = criterion.forward(inputs, target)
    dfdx = criterion.backward(inputs, target)
    fx1 = criterion.forward((inputs + eps), target)
    fx2 = criterion.forward((inputs - eps), target)
    central_diff_dfdx = (fx1 - fx2) / (2 * eps)
    err = np.max(np.abs(central_diff_dfdx - dfdx))
    return err < 1e-5


if __name__ == '__main__':
    import doctest
    doctest.testmod()
