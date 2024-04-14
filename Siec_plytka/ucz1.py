from Siec_plytka.dzialaj1 import calculate2, calculate1
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
    i=0
    Wafter = np.copy(Wprzed)
    beta = 5
    wspUcz = 0.1
    #liczba przykladow
    m = inputs.shape[0]
    ###print(f"liczba cech wejściowych ucz1: {m} ")

    for i in range(n):

        random_sample = np.random.randint(0,m)

        result, derivative = calculate1(Wafter[random_sample], inputs[random_sample])
        #result, derivative = calculate2(Wprzed, inputs)
        #####cost = T - result####


        #derivative of cost funciton with respect to W
        dj_dW = wspUcz * derivative
        #dj_dW = np.outer(cost, derivative)


        #change wages
        Wafter[random_sample] = Wafter[random_sample] + dj_dW


        ###print(f"Wafter: {Wafter}")
        ###rint(f"dj_dW: {dj_dW}")

    return Wafter



