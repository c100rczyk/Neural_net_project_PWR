import numpy as np

def calculate1(W, X):
    """
    Calculate the output

    :param W: matrix of wages
    :param X: vectors of inputs

    :return: vector of outputs
    """
    outputs = np.dot(W, X)
    beta = 5
    outputs_sigmoid = 1 / (1 + np.exp(-outputs * beta))

    derivative = beta * outputs_sigmoid * (1 - outputs_sigmoid)
    print(derivative)

    return outputs_sigmoid, derivative
