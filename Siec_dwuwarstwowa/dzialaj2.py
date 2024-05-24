import numpy as np

def calculate_outputs(W1, W2, X):
    """

    :param W1: macierz wag pierwszej warstwy sieci
    :param W2: macierz wag drugiej warstwy sieci
    :param X: wektor wejści do sieci (sygnał podany na wejście (sieci / warstwy 1)
    :return:
    Y1 : wektor wyjść watstwy pierwszej
    Y2 : wektor wyjść warstwy drugiej(sygnał na wyjściu sieci).
    """

    # np: W1 = [0.05 -0.21 -1]
    #      [-.61 0.01  -1]

    # np: X1 = [0 0 -1]

    beta = 5
    X1 = np.append(X, -1)               # dodanie bias=-1 do wejść
    U1 = np.dot(W1, X1)                 #wynik to wektor (o ilosci neuronów w pierwszej warstwie)
    Y1 = 1 / (1 + np.exp(-U1 * beta))   # funkcja aktywacji
    # Y1 = [1neuron , 2neuron] - wartości np [0.43, 0.98]


    # np: W2 = [0.15 -0.21 -1]
    Y1 = np.append(Y1, -1)
    U2 = np.dot(W2, Y1)     # jedna wartość na wyjściu
    Y2 = 1 / (1 + np.exp(-U2 * beta))    #wrzucam tą wartość do funkcji aktywacji

    return Y1, Y2   # return ( [x,y,z], skalar )

