from Siec_plytka.dzialaj1 import calculate1
import numpy as np

def ucz1(Wprzed, inputs, T, n):
    """
    Learn the network

    :param Wbefor: matrix of wages W before computing
    :param inputs
    :param T: requested outputs for each examples
    :param n: number of iterations
    :return: Wafter: matrix of wages W after training
    """

    w_init = Wprzed
    beta = 5
    #liczba przykladow
    m = inputs.shape[1]
    print(f"liczba cech wejściowych ucz1: {m} ")

    for i in range(n):

        random_sample = np.random.randint(0,m)

        result, derivative = calculate1(w_init, inputs[random_sample])
        cost = T[random_sample] - result


        #derivative of cost funciton with respect to W
        #dj_dW = 1/m * np.sum(derivative)        #_________
        dj_dW = np.outer(cost, derivative)

        #change wages
        w_init = w_init - 0.01 * dj_dW

    Wafter = w_init

    return Wafter



