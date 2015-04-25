# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:46:46 2015

@author: Sam
"""
import numpy as np


__all__ = ['StochasticGradient']


class StochasticGradient(object):
    
    def __init__(self, module, criterion):
        self.learning_rate = 0.01
        self.learning_rate_decay = 0
        self.max_iteration = 25
        self.shuffle_indices = True
        self.module = module
        self.criterion = criterion
        self.verbose = True

    def train(self, dataset):
        iteration = 1
        current_learning_rate = self.learning_rate
        module = self.module
        criterion = self.criterion
        shuffled_indices = np.random.shuffle(np.arange(dataset.size()))  # check
        if self.shuffle_indices is False:
            shuffled_indices = np.arange(dataset.size())
        
        print('# StochasticGradient: training')
        
        while True:
            current_error = 0
            
            for i in shuffled_indices:
                example = dataset[i]
                inputs = example[0]  # check
                target = example[1]
                current_error += criterion.forward(module.forward(inputs), target)
                module.update_grad_inputs(inputs, criterion.update_grad_inputs(module.output, target))
                module.acc_update_grad_params(inputs, criterion.grad_inputs, current_learning_rate)
                if self.hook_example:
                    self.hook_example(example)  #check
            
            current_error = current_error / dataset.size()
            
            if self.hook_iteration:
                self.hook_iteration(iteration, current_error)
            
            if self.verbose:
                print('# current error = ', current_error)
            
            iteration += 1
            current_learning_rate = self.learning_rate / (1 + iteration * self.learning_rate_decay)
            if self.max_iteration > 0 and iteration > self.max_iteration:
                print('# StochasticGradient: you have reached the maximum number of iterations')
                print('# training error = ', current_error)
                break

                