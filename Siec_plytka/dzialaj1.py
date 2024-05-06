#symuluje działanie sieci

import numpy as np

def calculate1(W, X):
    """
    Calculate the output

    :param W: matrix of wages

    |w11 w12 w13 w14 w15|
    |w21 w22 w23 w24 w25|
    |w31 w32 w33 w34 w35|

    :param X: vectors of inputs

    [ 4.    0.01  0.01 -1.   -1.5 ]

    :return: vector of outputs
    """


    outputs_sigmoid = 0
    #print(f"TEST_W: {W}")
    #print(f"TEST_X: {X}")
    outputs = np.dot(W, X)
    #print(f">>>>>>>>>>>>>>>{outputs} ")
    beta = 5
    outputs_sigmoid = 1 / (1 + np.exp(-outputs * beta))
    ###print(f"sigmoida:{outputs_sigmoid}")
    derivative = beta * outputs_sigmoid * (1 - outputs_sigmoid)
    return outputs_sigmoid


def calculate2(W,X):
    """
    :param W:
    [[ 0.38435499 -0.08775198  0.22202535  0.12875446  0.15566581]
    [ 0.65477037 -0.08333999  0.24986892  0.2818868   0.02709011]
    [ 0.53486141 -0.04132551  0.14515335  0.03183253 -0.05111972]]
    :param X:
    [2,0.2,0.3,0.1,0.05]
    :return:
    """
    outputs_sigmoid = [0.5,0.5,0.5]
    print(X[0])

    for i in range(3):
        outputs = np.dot(X,W[i])
        print(f"TEST: {outputs} ")
        beta = 5
        outputs_sigmoid[i] = 1 / (1 + np.exp(-outputs * beta))
        print(outputs_sigmoid[i])

        #derivative = beta * outputs_sigmoid * (1 - outputs_sigmoid)
        #print(f"pochodne: {derivative}")

    return outputs_sigmoid