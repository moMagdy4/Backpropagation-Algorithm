
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
# pre processing


def Pre_processing(data):
    data['gender'].fillna(data['gender'].mode()[0], inplace=True)
    data['gender'] = data.gender.map(dict(male=0, female=1))
    return data


def label_encoding(data):
    for i in range(len(data['species'])):
        if (data['species'][i] == 'Adelie'):
            data.loc[i, 'species'] = 0
        elif (data['species'][i] == 'Gentoo'):
            data.loc[i, 'species'] = 1
        else:
            data.loc[i, 'species'] = 2
    return data


def FeatureScalling(X):
    for column in X.columns:
        X[column] = (X[column] - X[column].min()) / \
            (X[column].max() - X[column].min())
    return X


def Split(data):
    X = data.drop('species', axis=1)
    X = FeatureScalling(X)
    y = data['species']
    # spliting data to train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, train_size=0.6, shuffle=True, random_state=10, stratify=y)

   # converting to numpy
    X_train = x_train.to_numpy()
    X_test = x_test.to_numpy()
    Y_train = y_train.to_numpy()
    Y_test = y_test.to_numpy()

    m = X_train.shape[0]  # number of training examples(60)

    Y_train = Y_train.reshape(m, 1)
    Y_test = Y_test.reshape(X_test.shape[0], 1)

    return X_train, X_test, Y_train, Y_test


def split2(data):
    X1 = data.drop('species', axis=1)

    y = data['species']
    X = FeatureScalling(X1)

    X = X.to_numpy()
    y = y.to_numpy()

    index_c1 = list(range(0, 50))
    index_c2 = list(range(50, 100))
    random.shuffle(index_c1)
    random.shuffle(index_c2)

    trian_indices = index_c1[:30]+index_c2[:30]
    test_indices = index_c1[30:]+index_c2[30:]

    random.shuffle(trian_indices)
    random.shuffle(test_indices)

    X_train = X[trian_indices, :]
    Y_train = y[trian_indices]
    X_test = X[test_indices, :]
    Y_test = y[test_indices]

    m = X_train.shape[0]  # number of training examples(60)
    Y_train = Y_train.reshape(m, 1)
    Y_test = Y_test.reshape(X_test.shape[0], 1)
    return X_train, X_test, Y_train, Y_test

# activation function


def activation_function(activation, z):
    if activation == 'sigmoid':

        g = 1/(1+np.exp(-z))
    elif activation == 'tanh':
        g = np.tanh(z)
    return g

# Model


def intialize_paramters(hidden_units, feature_num, bias):
    np.random.seed(42)
    parameters = {}

    for i in range(len(hidden_units)):

        parameters["w"+str(i+1)] = np.random.randn(hidden_units[i],
                                                   feature_num)*0.1

        if (bias == True):
            parameters["b"+str(i+1)] = np.random.randn(hidden_units[i], 1)*0.1

        feature_num = hidden_units[i]

    return parameters


def Forward_probagation(parameters, x, activation, bias):
    if (bias == True):
        L = len(parameters)//2
    if (bias == False):
        L = len(parameters)
    linear_cache = {}
    activation_cache = {}
    activation_cache['A0'] = x
    for i in range(L):
        if (bias == True):
            net = np.dot(parameters["w"+str(i+1)], x)+parameters['b'+str(i+1)]
        if (bias == False):
            net = np.dot(parameters["w"+str(i+1)], x)
        linear_cache['net'+str(i+1)] = net

        A = activation_function(activation, linear_cache['net'+str(i+1)])
        activation_cache["A"+str(i+1)] = A
        x = A

    return activation_cache


def back_probagation(parameters, activation_cache, y, activation, bias):
    derivatives = {}
    if (bias == True):
        L = len(parameters)//2
    if (bias == False):
        L = len(parameters)
    AL = activation_cache["A"+str(L)]
    if y == 0:
        n_y = np.array([1, 0, 0])

    elif y == 1:
        n_y = np.array([0, 1, 0])

    elif y == 2:
        n_y = np.array([0, 0, 1])

    n_y = np.array(n_y).reshape(3, 1)
    dA = n_y-AL

    for i in reversed(range(L)):

        if (activation == 'sigmoid'):
            g = AL*(1-AL)
        elif (activation == 'tanh'):
            g = 1-AL**2

        dz = dA*g

        dw = np.dot(dz, activation_cache["A"+str(i)].T)
        derivatives['dw'+str(i+1)] = dw
        if (bias == True):
            db = dz
            derivatives['db'+str(i+1)] = db

        dA = np.dot(parameters["w"+str(i+1)].T, dz)
        AL = activation_cache['A'+str(i)]

    return derivatives


def update(paramters, derivative, lr, bias):
    if (bias == True):
        L = len(paramters)//2
    if (bias == False):
        L = len(paramters)
    new_paramters = {}
    for i in range(L):
        new_paramters['w'+str(i+1)] = paramters['w'+str(i+1)
                                                ]+lr*derivative['dw'+str(i+1)]
        if (bias == True):
            new_paramters['b'+str(i+1)] = paramters['b'+str(i+1)
                                                    ]+lr*derivative['db'+str(i+1)]

    return new_paramters


def Mlp(hidden_units, x_train, y_train, activation, epocs, lr, b):
    x_train = x_train.T
    parameters = intialize_paramters(hidden_units, x_train.shape[0], b)
    for i in range(epocs):
        for j in range(x_train.shape[1]):
            x = x_train[:, j].reshape(x_train.shape[0], 1)
            y = y_train[j, 0]
            activation_cache = Forward_probagation(
                parameters, x, activation, b)
            drivatives = back_probagation(
                parameters, activation_cache, y, activation, b)
            parameters = update(parameters, drivatives, lr, b)

    return parameters


def confusionMatrix(paramters, x_test, y_test, activation, bias, TrainOrTest, x_train, y_train):
    if (TrainOrTest == True):
        x_test = x_test.T
        m = x_test.shape[1]  # number of training examples(40)
        n = x_test.shape[0]  # number of training features(2)
        preds = []
        y_list = []
        if (bias == True):
            L = len(paramters)//2
        if (bias == False):
            L = len(paramters)
        for i in range(m):
            activation_cache = Forward_probagation(
                paramters, x_test[:, i].reshape(n, 1), activation, bias)
            pred = activation_cache["A"+str(L)]
            pred = np.argmax(pred)
            preds.append(pred)
            y_list.append(y_test[i, 0])

        matrix = confusion_matrix(y_list, preds, labels=[0, 1, 2])
        acc = accuracy_score(y_list, preds)
    if (TrainOrTest == False):
        x_train = x_train.T
        m = x_train.shape[1]  # number of training examples(40)
        n = x_train.shape[0]  # number of training features(2)
        preds = []
        y_list = []
        if (bias == True):
            L = len(paramters)//2
        if (bias == False):
            L = len(paramters)
        for i in range(m):
            activation_cache = Forward_probagation(
                paramters, x_train[:, i].reshape(n, 1), activation, bias)
            pred = activation_cache["A"+str(L)]
            pred = np.argmax(pred)
            preds.append(pred)
            y_list.append(y_train[i, 0])

        matrix = confusion_matrix(y_list, preds, labels=[0, 1, 2])
        acc = accuracy_score(y_list, preds)
    return matrix, acc
