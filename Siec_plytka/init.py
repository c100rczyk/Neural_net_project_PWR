import numpy as np

def init1(S,K):
    """
    Creating network - matrix of wages and fullfilled wages with random numbers
    from -0.1 to 0.1

    :param S: number of inputs to network
    :param K: number of neurons in layer
    :return: W - matrix of wages in network
    """
    w = np.zeros((K, S))
    for i in range(K):
        for j in range(S):
            w[i, j] = np.random.uniform(-0.1, 0.1)

    return w





