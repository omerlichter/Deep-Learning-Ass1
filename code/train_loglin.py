import loglinear as ll
import numpy as np
import random
import utils

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # Should return a numpy vector of features.
    vec = np.zeros(600)
    for feat in features:
        if utils.F2I.has_key(feat):
            vec[utils.F2I.get(feat)] += 1
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y = utils.L2I.get(label)
        y_hat = ll.predict(x, params)
        if y == y_hat:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I.get(label)   # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss

            # update the parameters according to the gradients
            # and the learning rate.
            W, b = params
            gW, gb = grads
            W -= learning_rate * gW
            b -= learning_rate * gb

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':

    TRAIN_FILE = "data/train"
    DEV_FILE = "data/dev"

    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    in_dim = 600
    out_dim = 6
    num_iterations = 10
    learning_rate = 0.1

    train_data = utils.TRAIN
    dev_data = utils.DEV
    # ...
   
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

