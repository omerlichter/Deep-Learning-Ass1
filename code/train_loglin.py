import mlpn as mln
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
        y_hat = mln.predict(x, params)
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
            loss, grads = mln.loss_and_gradients(x,y,params)
            cum_loss += loss

            # update the parameters according to the gradients
            # and the learning rate.
            for i, param in enumerate(params):
                param -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


def test(test_data, result_file, params):

    file = open(result_file, "w")

    for label, features in test_data:
        x = feats_to_vec(features)              # convert features to a vector.
        y = utils.I2L.get(mln.predict(x, params))   # predict
        file.write("" + y + "\n")
    file.close()

if __name__ == '__main__':

    result_file = "../data/result"

    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    num_iterations = 10
    learning_rate = 0.1

    train_data = utils.TRAIN
    dev_data = utils.DEV
    test_data = utils.TEST

    params = mln.create_classifier([600, 100, 30, 6])
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    test(test_data, result_file , params)

