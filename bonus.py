import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from scipy.sparse import *
np.random.seed(42)


def activation_function(activation, z):
    if activation == 'sigmoid':

        g = 1/(1+np.exp(-z))
    elif activation == 'tanh':
        g = np.tanh(z)
    return g


def Y_transform(y):
    y_train = []
    for i in range(len(y)):
        if y[i] == 0:
            n_y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif y[i] == 1:
            n_y = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        elif y[i] == 2:
            n_y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif y[i] == 3:
            n_y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif y[i] == 4:
            n_y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif y[i] == 5:
            n_y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif y[i] == 6:
            n_y = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif y[i] == 7:
            n_y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif y[i] == 8:
            n_y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif y[i] == 9:
            n_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        y_train.append(n_y)

    y_train = np.array(y_train)
    return y_train


def intialize_paramters(hidden_units, feature_num):
    np.random.seed(42)
    parameters = {}

    for i in range(len(hidden_units)):

        parameters["w"+str(i+1)] = np.random.randn(hidden_units[i],
                                                   feature_num)*0.1
        parameters["b"+str(i+1)] = np.random.randn(hidden_units[i], 1)*0.1
        feature_num = hidden_units[i]

    return parameters


def Forward_probagation(parameters, x, activation):
    L = len(parameters)//2
    linear_cache = {}
    activation_cache = {}
    activation_cache['A0'] = x
    for i in range(L):
        net = np.dot(parameters["w"+str(i+1)], x)+parameters['b'+str(i+1)]
        linear_cache['net'+str(i+1)] = net

        A = activation_function(activation, linear_cache['net'+str(i+1)])
        activation_cache["A"+str(i+1)] = A
        x = A

    return activation_cache


def back_probagation(parameters, activation_cache, y, activation):
    derivatives = {}
    L = len(parameters)//2
    AL = activation_cache["A"+str(L)]
    m = y.shape[1]

    dA = y-AL
    for i in reversed(range(L)):

        if (activation == 'sigmoid'):
            g = AL*(1-AL)
        elif (activation == 'tanh'):
            g = 1-AL**2

        dz = dA*g

        dw = np.dot(dz, activation_cache["A"+str(i)].T)
        derivatives['dw'+str(i+1)] = (1/m)*dw

        db = dz
        derivatives['db'+str(i+1)] = (1/m)*np.sum(db, axis=1, keepdims=True)

        dA = np.dot(parameters["w"+str(i+1)].T, dz)
        AL = activation_cache['A'+str(i)]

    return derivatives


def update(paramters, derivative, lr):
    L = len(paramters)//2
    new_paramters = {}
    for i in range(L):
        new_paramters['w'+str(i+1)] = paramters['w'+str(i+1)
                                                ]+lr*derivative['dw'+str(i+1)]
        new_paramters['b'+str(i+1)] = paramters['b'+str(i+1)
                                                ]+lr*derivative['db'+str(i+1)]

    return new_paramters


def Mlp(hidden_units, x_train, y_train, activation, epocs, lr):
    x_train = x_train.T
    y_train = y_train.T
    parameters = intialize_paramters(hidden_units, x_train.shape[0])

    for i in range(epocs):
        activation_cache = Forward_probagation(parameters, x_train, activation)
        drivatives = back_probagation(
            parameters, activation_cache, y_train, activation)

        parameters = update(parameters, drivatives, lr)

    return parameters


def confusionMatrix(paramters, x_test, y_test, activation):

    x_test = x_test.T
    y_test = y_test.T
    m = x_test.shape[1]  # number of training examples(40)
    n = x_test.shape[0]  # number of training features(2)

    L = len(paramters)//2

    activation_cache = Forward_probagation(paramters, x_test, activation)
    pred = activation_cache["A"+str(L)]
    pred = np.argmax(pred, axis=0)

    y_test = np.argmax(y_test, axis=0)

    matrix = confusion_matrix(y_test, pred, labels=[
                              0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    acc = accuracy_score(y_test, pred)
    return matrix, acc

# pre processing


def remove_con_values(pixels_data):
    reduction_pixels = pixels_data.loc[:]
    # For black
    dropped_black_pixels = []
    for col in pixels_data:
        if reduction_pixels[col].max() == 0:
            reduction_pixels.drop(columns=[col], inplace=True)
            dropped_black_pixels.append(col)

    # for white
    dropped_white_pixels = []
    for col in reduction_pixels:
        if reduction_pixels[col].min() == 255:
            reduction_pixels.drop(columns=[col], inplace=True)
            dropped_white_pixels.append(col)

    return reduction_pixels
