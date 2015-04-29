"""
re-implements nn.module in python

@author: Sam
"""
from copy import deepcopy
from tensor import Tensor

__all__ = ['Module', 'Container', 'Sequential']


class Module(object):
    """Module is an abstract class which defines fundamental methods
    necessary for a training a neural network. Modules are serializable.

    Modules contain two states variables: output and grad_inputs.

    Output contains the output of the module, computed with the last
    call of forward(input).

    Grad_inputs contains the gradients with respect to the inputs of the
    module, computed with the last call of
    update_grad_inputs(input, grad_output).

    These state variables are useful objects if one wants to check the
    guts of a Module. The object pointer is never supposed to change.
    However, its contents (including its size if it is a Tensor) are
    supposed to change.

    In general state variables are Tensors. Please, refer to each module
    specification for further information.
    """
    def __init__(self):
        self.grad_inputs = []#Tensor()
        self.output = []#Tensor()

    def parameters(self):
        """This function should returns two values. One for the
        learnable parameters (weights) and another for the gradients of
        the energy w.r.t. to the learnable parameters (grad_weights).

        Custom modules should override this function if they use
        learnable parameters that are stored in tensors.
        """
        if self.weight and self.bias:
            return (self.weight, self.bias), (self.grad_weight, self.grad_bias)
        elif self.weight:
            return self.weight, self.grad_weight
        elif self.bias:
            return self.bias, self.grad_bias
        else:
            return None, None

    def update_output(self, inputs):
        """Computes the output using the current parameter set of the
        class and input. This function returns the result which is
        stored in the output field.
        """
        return self.output

    def forward(self, inputs):
        """Takes an input object, and computes the corresponding output
        of the module. In general input and output are Tensors. Please,
        refer to each module specification for further information.

        After a forward(), the ouput state variable should have been
        updated to the new value.

        It is not advised to override this function. Instead, one should
        implement update_output(input) function. The forward module in
        the abstract parent class Module will call update_output(input).
        """
        return self.update_output(inputs)

    def backward(self, inputs, grad_output, scale=1):
        """Performs a backpropagation step through the module, with
        respect to the given input. In general this method makes the
        assumption forward(input) has been called before, with the same
        input. This is necessary for optimization reasons. If you do not
        respect this rule, backward() will compute incorrect gradients.

        In general input and gradOutput and gradInput are Tensors.
        Please, refer to each module specification for further
        information.

        A backpropagation step consists of computing two kind of
        gradients at input given grad_output (gradients with respect to
        the output of the module). This function simply performs this
        task using two function calls.

        A function call to update_grad_inputs(input, grad_output).
        A function call to acc_grad_params(input,grad_output,scale).
        
        It is not advised to override this function call in custom
        classes. It is better to override
        update_grad_inputs(input, grad_output) and
        acc_grad_params(input, grad_output, scale) functions.
        """
        self.update_grad_inputs(inputs, grad_output)
        self.acc_grad_params(inputs, grad_output, scale)
        return self.grad_inputs

    def backward_update(self, inputs, grad_output, lr):
        self.update_grad_inputs(inputs, grad_output)
        self.acc_update_grad_params(inputs, grad_output, lr)
        return self.grad_inputs

    def update_grad_inputs(self, inputs, grad_output):
        """Computing the gradient of the module with respect to its own
        input. This is returned in grad_inputs. Also, the grad_inputs
        state variable is updated accordingly.
        """
        return self.grad_inputs

    def acc_grad_params(self, inputs, grad_output, scale):
        """Computing the gradient of the module with respect to its own
        parameters. Many modules do not perform this step as they do not
        have any parameters. The state variable name for the parameters
        is module dependent. The module is expected to accumulate the
        gradients with respect to the parameters in some variable.

        Scale is a scale factor that is multiplied with the
        grad_params before being accumulated.

        Zeroing this accumulation is achieved with zero_grad_params()
        and updating the parameters according to this accumulation is
        done with update_params().
        """
        pass

    def acc_update_grad_params(self, inputs, grad_output, lr):
        """This is a convenience module that performs two functions at
        once. Calculates and accumulates the gradients with respect to
        the weights after mutltiplying with the negative learning
        rate. Performing these two operations at once is more
        performance efficient and it might be advantageous in certain
        situations.

        Keep in mind that, this function uses a simple trick to achieve
        its goal and it might not be valid for a custom module.

        Also note that compared to acc_grad_parames(), the gradients are
        not retained for future use.
        
        The gradients are accumulated directly into weights. This
        assumption may not be true for a module that computes a
        nonlinear operation.
        """
        grad_weight = self.grad_weight
        grad_bias = self.grad_bias
        self.grad_weight = self.weight
        self.grad_bias = self.bias
        self.acc_grad_params(inputs, grad_output, -lr)
        self.grad_weight = grad_weight
        self.grad_bias = grad_bias

    def shared_acc_update_grad_params(self, inputs, grad_output, lr):
        if self.parameters():
            self.zero_grad_params()
            self.acc_grad_params(inputs, grad_output, 1)
            self.update_params(lr)

    def zero_grad_params(self):
        """If the module has parameters, this will zero the accumulation
        of the gradients with respect to these parameters, accumulated
        through acc_grad_params(input, grad_output, scale) calls.
        Otherwise, it does nothing.
        """
        _, grad_params = self.parameters()
        if grad_params:
            grad_params.fill(0)

    def update_params(self, learning_rate):
        """If the module has parameters, this will update these
        parameters, according to the accumulation of the gradients with
        respect to these parameters, accumulated through backward()
        calls.

        The update is basically:

        parameters = parameters - learning_rate * gradients_wrt_params
        
        If the module does not have parameters, it does nothing.
        """
        params, grad_params = self.parameters()
        if params:
            params = params - learning_rate * grad_params

    def training(self):
        """This sets the mode of the Module (or sub-modules) to train =
        True. This is useful for modules like Dropout that have a
        different behaviour during training vs evaluation.
        """
        self.train = True

    def evaluate(self):
        """This sets the mode of the Module (or sub-modules) to train =
        False. This is useful for modules like Dropout that have a
        different behaviour during training vs evaluation.
        """
        self.train = False

    def share(self, mlp, *args):
        """
        """
        for arg in args:
            if self.arg != None:
                print(self.arg)

    def clone(self):
        """Creates a deep copy of (i.e. not just a pointer to) the
        module, including the current state of its parameters (e.g.
        weight, biases etc., if any).
        """
        return deepcopy(self)

    def is_type(self):
        return type(self)

    def reset():
        pass

    def get_params(self):
        """This function returns two flattened tensors. One for the
        learnable parameters and another for the gradients of the energy
        w.r.t. to the learnable parameters.

        Custom modules should not override this function. They should
        instead override parameters(...) which is, in turn, called by
        the present function.

        This function will go over all the weights and grad_weights and
        make them view into a single tensor (one for weights and one for
        grad_weights). Since every weight and grad_weight is changed,
        this function should be called only once on a given network.
        """
        params, grad_params = self.parameters()

        if params is None:
            return Tensor()

        return params.ravel(), grad_params.ravel()

    def __call__(self, inputs, grad_output=None):
        self.forward(inputs)
        if grad_output:
            self.backward(inputs, grad_output)
            return self.output, self.grad_inputs
        else:
            return self.output


class Container(Module):
    
    def __init__(self):
        Module.__init__(self)
        self.modules = []

    def add(self, module):
        self.modules.extend(module)

    def get(self, index):
        return self.modules[index]

    def size(self):
        return len(self.modules)

    def zero_grad_params(self):
        for mod in self.modules:
            mod.zero_grad_params()

    def update_params(self, learning_rate):
        for mod in self.modules:
            mod.update_params(learning_rate)

    def training(self):
        for mod in self.modules:
            mod.training()

    def evaluate(self):
        for mod in self.modules:
            mod.evaluate()

    def share(self):
        pass

    def reset(self, std):
        for mod in self.modules:
            mod.reset(std)

    def parameters(self):
        weight = []
        grad_weight = []
        for mod in self.modules:
            mod_weight, mod_grad_weight = mod.parameters()
            if mod_weight:
                weight.extend(mod_weight)
                grad_weight.extend(mod_grad_weight)
        return weight, grad_weight  #  maybe these should be self.weight = weight


class Sequential(Container):
    
    def add(self, module):
        if len(self.modules) == 0:
            self.grad_inputs = module.grad_inputs
        self.modules.extend(module)
        self.output = module.output

    def insert(self, module, index):
        assert index <= len(self.modules + 1), 'index should be contiguous'
        self.modules.insert(index, module)
        self.output = self.modules[-1].output
        self.grad_inputs = self.modules[0].grad_inputs

    def update_output(self, inputs):
        current_output = inputs
        for mod in self.modules:
            current_output = mod.update_output(current_output)
        self.output = current_output
        return current_output  # do we need a return here

    def update_grad_inputs(self, inputs, grad_output):
        current_grad_out = grad_output
        current_mod = self.modules[-1]
        for mod in reversed(self.modules[:-1]):
            current_grad_out = current_mod.update_grad_inputs(mod.output, current_grad_out)
            current_mod = mod
        current_grad_out = current_mod.update_grad_inputs(inputs, current_grad_out)
        self.grad_inputs = current_grad_out
        return current_grad_out  # do we need a return here

    def acc_grad_params(self, inputs, grad_output, scale):
        if scale is None:
            scale = 1
        current_grad_out = grad_output
        current_mod = self.modules[-1]
        for mod in reversed(self.modules[:-1]):
            current_mod.acc_grad_params(mod.output, current_grad_out, scale)
            current_grad_out = current_mod.grad_inputs
            current_mod = mod
        current_mod.acc_grad_params(inputs, current_grad_out, scale)

    def acc_update_grad_params(self, inputs, grad_output, lr):
        current_grad_out = grad_output
        current_mod = self.modules[-1]
        for mod in reversed(self.modules[:-1]):
            current_mod.acc_update_grad_params(mod.output, current_grad_out, lr)
            current_grad_out = current_mod.grad_inputs
            current_mod = mod
        current_mod.acc_update_grad_params(inputs, current_grad_out, lr)
