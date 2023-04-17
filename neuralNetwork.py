"""
Pablo Valencia
A01700912
"""

import numpy as np


def init_parameters(neurons):

    W1 = np.random.randn(neurons[1], neurons[0]) * 0.01
    b1 = np.zeros((neurons[1], 1))

    W2 = np.random.randn(neurons[2], neurons[1]) * 0.01
    b2 = np.zeros((neurons[2], 1))

    W3 = np.random.randn(neurons[3], neurons[1]) * 0.01
    b3 = np.zeros((neurons[3], 1))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}


def forward(X, parameters, activation_functions):
    Z1 = parameters['W1'] @ X + parameters['b1']
    A1 = activation_functions[0](Z1)
    Z2 = parameters['W2'] @ A1 + parameters['b2']
    A2 = activation_functions[1](Z2)
    Z3 = parameters['W3'] @ A2 + parameters['b3']

    return Z3, Z2, Z1, A2, A1


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_scores = np.exp(x)
    sum_exp_scores = np.sum(exp_scores, axis=0)
    predictions = exp_scores / sum_exp_scores
    return predictions


def x_entropy(scores, y, batch_size):
    predictions = softmax(scores)
    y_hat = predictions[y.squeeze(), np.arange(batch_size)]
    cost = np.sum(-np.log(y_hat)) / batch_size

    return predictions, cost


def backward(predictions, x, y, Z2, A2, Z1, A1, parameters, batch_size):

    predictions[y.squeeze(), np.arange(len(x))] -= 1
    dz3 = predictions.copy()

    dW3 = dz3 @ A2.T / batch_size
    db3 = np.sum(dz3, axis=1, keepdims=True) / batch_size
    da2 = parameters['W3'].T @ dz3

    dz2 = da2 * (Z2 > 0)
    dW2 = dz2 @ A1.T / batch_size
    db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
    da1 = parameters['W2'].T @ dz2

    dz1 = da1 * (Z1 > 0)
    dW1 = dz1 @ x / batch_size
    db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    return grads
