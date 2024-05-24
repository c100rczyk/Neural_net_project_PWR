import numpy as np

def init2(S, K1, K2):
    """

    :param S: liczba wejść do sieci
    :param K1: liczba neuronów w pierwszej warstwie
    :param K2: liczba neuronów w drugiej warstwie
    :return: W1, W2  macierze wag warstw
    """

    i=j=0

    w1 = np.zeros((K1 , S+1))
    w2 = np.zeros((K2, K1+1))
    """
    [ [wagi do pierwszego neuronu],
      [wagi do drugiego neuronu],
      [wagi do k-tego neuronu]]
    """

    for i in range(K1):
        for j in range(S+1):
            w1[i,j] = np.random.uniform(-0.1, 0.1)
        #w1[i,j+1] = -1

    for i in range(K2):
        for j in range(K1+1):
            w2[i,j] = np.random.uniform(-0.1, 0.1)
        #w2[i,j+1] = -1


    return w1, w2

    # w1 = np.random.randn(K1, S + 1) * np.sqrt(2 / (S + 1))
    # w2 = np.random.randn(K2, K1 + 1) * np.sqrt(2 / (K1 + 1))
    #
    # return w1, w2

    #return np.random.rand( K1,S + 1) * 0.2 - 0.1, np.random.rand(K2, K1+1) * 0.2 - 0.1