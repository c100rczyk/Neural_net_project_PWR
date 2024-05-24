from Siec_dwuwarstwowa.dzialaj2 import calculate_outputs
import numpy as np

def ucz2(W1przed, W2przed, examples, True_value, n_iter, error):
    """
    Training wages of layers to get optimal matrixes of wages
    :param W1przed: matrix of wages for first layer
    :param W2przed: matrix of wages for second layer
    :param examples: examples to train
    :param True_value:
    :param n_iter: how main times to update

    :return: W1_optimal , W2_optimal, MSE
    """
    wspUcz = 0.01
    beta = 5
    momentum = 0.9
    W1 = W1przed.copy()
    W2 = W2przed.copy()
    m = examples.shape[1]
    wynik = calculate_outputs(W1przed, W2przed, examples[:,0])
    mse_history=[]

    v_W1 = np.zeros_like(W1)
    v_W2 = np.zeros_like(W2)



    for i in range(n_iter):


        random_sample = np.random.randint(0,m)
        example = examples[:,random_sample]

        # Forward Propagation
        Y1, Y2 = calculate_outputs(W1, W2, example)

        # Backpropagation
        # wyznaczenie błędów w każdej z warstw
        cost = True_value[random_sample] - Y2
        derivative_2warstwa = cost * beta * Y2 * (1-Y2)
        #Y1_aug = np.append(Y1, -1)
        dj_dW2 = wspUcz * np.outer(derivative_2warstwa, Y1)

        error_hidden_layer = np.dot(W2.T , derivative_2warstwa)[:-1]    # bias nie ulega aktualizacji

        derivative_1warstwa = error_hidden_layer * beta * Y1[:-1] * ( 1  - Y1[:-1])
        example_aug = np.append(example, -1)
        dj_dW1 = wspUcz * np.outer(derivative_1warstwa,example_aug)
        v_W1 = dj_dW1 + momentum * v_W1
        v_W2 = dj_dW2 + momentum * v_W2

        W1 += v_W1
        W2 += v_W2

        mse = np.mean((True_value[random_sample] - calculate_outputs(W1,W2, example)[1])**2)
        mse_history.append(mse)

        #Adaptacyjny learning rate
        #p = 0.5
        #wspUcz = wspUcz / ((i+1) **2)

        if(mse < error):
            break

    return W1, W2, mse_history





def ucz2TEST(W1przed, W2przed, examples, True_value, n_iter):
    """
    Training wages of layers to get optimal matrixes of wages
    :param W1przed: matrix of wages for first layer
    :param W2przed: matrix of wages for second layer
    :param examples: examples to train
    :param True_value:
    :param n_iter: how main times to update

    :return: W1_optimal , W2_optimal, MSE
    """
    wspUcz = 0.01
    beta = 5
    momentum = 0.9
    W1 = W1przed.copy()
    W2 = W2przed.copy()
    m = examples.shape[1]
    wynik = calculate_outputs(W1przed, W2przed, examples[:,0])
    mse_history=[]

    v_W1 = np.zeros_like(W1)
    v_W2 = np.zeros_like(W2)
    for i in range(n_iter):


        random_sample = np.random.randint(0,m)
        example = examples[:,random_sample]


        # Forward Propagation
        Y1, Y2 = calculate_outputs(W1, W2, example)
        # Backpropagation
        # wyznaczenie błędów w każdej z warstw
        cost = True_value[random_sample] - Y2
        derivative_2warstwa = cost * beta * Y2 * (1-Y2)
        dj_dW2 = wspUcz * np.outer(derivative_2warstwa, Y1)

        print(f"derivative2_warstwa: {derivative_2warstwa}")
        #Y1_aug = np.append(Y1, -1)
        #dj_dW2 = wspUcz * np.outer(derivative_2warstwa, Y1)
        print(f"dj_dW2 {dj_dW2}")
        print(f"W2 {W2}")


        print(f"W2.T{W2.T}")
        error_hidden_layer = np.dot(W2.T , derivative_2warstwa)[:-1]    # bias nie ulega aktualizacji
        print(f"error_hidden_layer {error_hidden_layer}")   # 3 wartości błędów?

        derivative_1warstwa = error_hidden_layer * beta * Y1[:-1] * ( 1  - Y1[:-1])
        print(f"derivative1_warstwa: {derivative_1warstwa}")
        example_aug = np.append(example, -1)
        dj_dW1 = wspUcz * np.outer(derivative_1warstwa,example_aug)
        print(f"dj_dW1 {dj_dW1}")
        v_W1 = dj_dW1 + momentum * v_W1
        v_W2 = dj_dW2 + momentum * v_W2

        W1 += v_W1
        W2 += v_W2

        mse = np.mean((True_value[random_sample] - calculate_outputs(W1,W2, example)[1])**2)
        mse_history.append(mse)

    return W1, W2, mse_history




