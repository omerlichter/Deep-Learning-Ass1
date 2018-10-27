import mlp1 as ml
import loglinear as ll
import numpy as np
import random
import utils
import xor_data

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = features
        y = label
        y_hat = ml.predict(x, params)
        if y == y_hat:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = ml.loss_and_gradients(x, y, params)
            cum_loss += loss

            # update the parameters according to the gradients
            # and the learning rate.
            W, b, U, b_tag = params
            gW, gb, gU, gb_tag = grads
            W -= learning_rate * gW
            b -= learning_rate * gb
            U -= learning_rate * gU
            b_tag -= learning_rate * gb_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        print I, train_loss, train_accuracy
    return params


if __name__ == '__main__':
    TRAIN_FILE = "data/train"
    DEV_FILE = "data/dev"

    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    in_dim = 2
    hid_dim = 40
    out_dim = 2
    num_iterations = 10
    learning_rate = 0.1

    train_data = xor_data.data
    # ...

    params = ml.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, num_iterations, learning_rate, params)


    print "prediction of xor: ", ml.predict([1, 1], params)

