import numpy as np

def calculate1(W, X):
    """
    Calculate the output

    :param W: matrix of wages
    :param X: vectors of inputs

    :return: vector of outputs
    """


    outputs_sigmoid = 0
    ###print(f"TEST_W: {W}")
    ###print(f"TEST_X: {X}")
    outputs = np.dot(W, X)
    beta = 5
    outputs_sigmoid = 1 / (1 + np.exp(-outputs * beta))
    ###print(f"sigmoida:{outputs_sigmoid}")
    derivative = beta * outputs_sigmoid * (1 - outputs_sigmoid)
    ###print(f"pochodne: {derivative}")
    return outputs_sigmoid, derivative


def calculate2(W,X):

    outputs_sigmoid = [0.5,0.5,0.5]
    print(X[0])

    for i in range(X.shape[0]):
        outputs = np.dot(W, X[i])
        print(f"TEST: {outputs} ")
        beta = 5
        outputs_sigmoid = 1 / (1 + np.exp(-outputs * beta))

        #derivative = beta * outputs_sigmoid * (1 - outputs_sigmoid)
        #print(f"pochodne: {derivative}")

    return outputs_sigmoid