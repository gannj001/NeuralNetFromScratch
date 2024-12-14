

class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    # compute the output of a layer for a given input
    def forward_propagation(self, input_data):
        raise NotImplementedError

    # computes derivative of dE and dX for a given dE/dY and update the parameters if valid
    # learning_rate will be implemented through gradient descent instead of a formal optimizer
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    