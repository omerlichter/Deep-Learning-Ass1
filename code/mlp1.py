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
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

def tanh_diff(x):
    """
    compute the tanh_diff vector
    x: a n-dim vector (numpy array)
    return: a n-dim vector (numpy array) of tanh_diff values
    """
    return 1 - pow(tanh(x), 2)

def classifier_output(x, params):
    W, b, U, b_tag = params
    h = tanh(np.dot(x, W) + b)
    probs = softmax(np.dot(h, U) + b_tag)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    h = tanh(np.dot(x, W) + b)
    y_hat = softmax(np.dot(h, U) + b_tag)
    y_real = np.zeros(y_hat.shape)
    y_real[y] = 1
    loss = -np.log(y_hat[y])
    g_loss_y_hat = -(y_real - y_hat)

    ### first layer
    gU = np.outer(h, g_loss_y_hat)
    gb_tag = g_loss_y_hat

    ### second layer
    g_loss_h = np.dot(U, g_loss_y_hat) * tanh_diff(h)
    gW = np.outer(x, g_loss_h)
    gb = g_loss_h

    return loss,[gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)
    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = tanh(np.array([0.234, 0.8787]))
    print test1


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b,U,b_tag = create_classifier(3,4,5)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


