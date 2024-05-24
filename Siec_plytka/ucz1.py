# uczy sieć na zadanym ciągu uczącym

from Siec_plytka.dzialaj1 import calculate2, calculate1
import numpy as np

def ucz1(Wprzed, examples, True_value, n_iter, min_error):
    """
    Learn the network

    :param Wprzed: matrix of wages W before computing
    :param examples
    [[ 4.    2.   -1.  ]
    [ 0.01 -1.    3.5 ]
    [ 0.01  2.    0.01]
    [-1.    2.5  -2.  ]
    [-1.5   2.    1.5 ]]
    :param True_value: requested outputs for each examples
    :param n_iter: number of iterations
    :return: Wafter: matrix of wages W after training
    """
    i=0
    Wafter = np.copy(Wprzed)
    beta = 5
    wspUcz = 0.1
    # liczba przykladow
    m = examples.shape[0]
    ###print(f"liczba cech wejściowych ucz1: {m} ")
    liczba_wyjsc = len(True_value[0])
    MSE = np.zeros(1000)
    ucz_dalej = True

    for i in range(n_iter):
        random_sample = np.random.randint(0,m)

        result = calculate1(Wafter, examples[random_sample])  #Wafter[random_sample]
        print(f"result {result}")
        # result, derivative = calculate2(Wprzed, inputs)
        cost = True_value[random_sample] - result
        derivative = cost * beta * result * (1-result)
        print(f"cost: {cost}")


        # derivative of cost funciton with respect to W
        #dj_dW =      #np.outer(cost , derivative)   # vector
        #print(f"dj_dW {dj_dW}")
        # dj_dW = np.outer(cost, derivative)
        print("TESTY")
        print(wspUcz)
        print(Wafter)
        print("Koniec TESTOW")
        #change wages
        print("OBLICZANIE dj_dW:")
        print(f"expamples[random_sample]: {examples[random_sample]}")
        print(f"derivative: {derivative}")
        print("Koniec")

        dj_dW = wspUcz * np.outer(derivative, examples[random_sample])
        print(f"dj_dW: {dj_dW}")
        Wafter = Wafter + dj_dW
        print("_____________________________")
        print(f"cost {cost}")
        print(f"cost.T {cost[:, np.newaxis]}")
        print(0.5* 1/liczba_wyjsc * np.dot(cost , cost[:, np.newaxis]) )
        MSE[i] = 0.5* 1/liczba_wyjsc * np.dot(cost , cost[:, np.newaxis])
        if(MSE[i] <= min_error):
            i=1000

        ###print(f"Wafter: {Wafter}")
        ###rint(f"dj_dW: {dj_dW}")

    return Wafter, MSE



