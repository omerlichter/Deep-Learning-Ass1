import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    e_x = np.exp(x - np.max(x))
    x = e_x / e_x.sum()
    return x


def tanh(x):
    """
    compute the tanh vector
    x: a n-dim vector (numpy array)
    return: a n-dim vector (numpy array) of tanh values
    """
    return np.tanh(x)

def tanh_diff(x):
    """
    compute the tanh_diff vector
    x: a n-dim vector (numpy array)
    return: a n-dim vector (numpy array) of tanh_diff values
    """
    return 1 - pow(tanh(x), 2)


def classifier_output(x, params):
    h_i = x
    for W_i, b_i in zip(params[0:-2:2], params[1:-1:2]):
        h_i = tanh(np.dot(h_i, W_i) + b_i)
    probs = softmax(np.dot(h_i, params[-2]) + params[-1])
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """

    h = [x]
    for W_i, b_i in zip(params[0:-2:2], params[1:-1:2]):
        h.append(tanh(np.dot(h[-1], W_i) + b_i))
    y_hat = softmax(np.dot(h[-1], params[-2]) + params[-1])
    y_real = np.zeros(y_hat.shape)
    y_real[y] = 1

    loss = -np.log(y_hat[y])

    grads = []
    ### gradient of loss by y_hat
    g_until_now = -(y_real - y_hat)


    for i, (W_i, b_i) in enumerate(zip(params[-2::-2], params[-1::-2])):
        g_b_i = np.copy(g_until_now)
        g_w_i = np.outer(h[-i - 1], g_until_now)
        grads.append(g_b_i)
        grads.append(g_w_i)
        g_until_now = np.dot(W_i, g_until_now) * tanh_diff(h[-i - 1])

    grads = list(reversed(grads))
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for dim_1, dim_2 in zip(dims, dims[1:]):
        W_i = np.zeros((dim_1, dim_2))
        b_i = np.zeros(dim_2)
        eps = np.sqrt(6) / (np.sqrt(W_i.shape[0] + W_i.shape[1]))
        W_i = np.random.uniform(-eps, eps, (W_i.shape[0], W_i.shape[1]))
        eps = np.sqrt(6) / (np.sqrt(b_i.shape[0]))
        b_i = np.random.uniform(-eps, eps, b_i.shape[0])
        params.append(W_i)
        params.append(b_i)


    return params

