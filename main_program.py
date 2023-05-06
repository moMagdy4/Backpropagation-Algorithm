
from tkinter import *
from tkinter.ttk import Separator
from tkinter import ttk
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import Helper_functions as model
import bonus as BONUS
from sklearn.decomposition import PCA
np.random.seed(42)


def string_to_list(no_of_neurons, hiddenlayers, Y_train):
    # number of hidden units in output layer
    class_num = len(np.unique(Y_train))
    class_num
    new_np = no_of_neurons.split(",")
    new1 = []
    for x in range(hiddenlayers):
        new1.append(int(new_np[x]))
    new1.append(class_num)

    print("list")
    print(new1)
    return new1


def bonus():
    data_minist_train = pd.read_csv('mnist_train.csv')
    data_minist_test = pd.read_csv('mnist_test.csv')

    x_train_mnist = data_minist_train.drop('label', axis=1)
    y_train_mnist = data_minist_train['label']
    pca = PCA(n_components=196)
    # xtrain = BONUS.remove_con_values(x_train_mnist)  # xtrain

    x_test_mnist = data_minist_test.drop('label', axis=1)
    y_test_mnist = data_minist_test['label']

    # xtest = BONUS.remove_con_values(x_test_mnist)  # xtest

    xtrain = pca.fit_transform(x_train_mnist)
    xtest = pca.transform(x_test_mnist)
    ytrain = BONUS.Y_transform(y_train_mnist)
    ytest = BONUS.Y_transform(y_test_mnist)

    act_fun = AC_Value.get()

    hidden = hiddenlayers_TextField.get()
    hidden_int = hiddenlayers_TextField.getint(hidden)

    no_neurons = neuorns_hiddenlayers_TextField.get()
    no_neurons_int = string_to_list(no_neurons, hidden_int, y_train_mnist)

    le_str = learningRate_TextField.get()
    le_rate = learningRate_TextField.getdouble(le_str)

    epochs_str = number_Of_Epochs_TextField.get()
    epochs = number_Of_Epochs_TextField.getint(epochs_str)

    bias = biasCheckBox.get()
    test_on = TestCheckBox.get()
    param = BONUS.Mlp(no_neurons_int, xtrain, ytrain, act_fun, epochs, le_rate)
    if (test_on == True):
        mat, acc = BONUS.confusionMatrix(param, xtest, ytest, act_fun)
    if (test_on == False):
        mat, acc = BONUS.confusionMatrix(param, xtrain, ytrain, act_fun)

    show_acc(mat, acc, test_on)


def Data_from_gui_to_Model():
    data = pd.read_csv('penguins.csv')
    # data
    processing_data = model.Pre_processing(data)
    processing_labeld_data = model.label_encoding(processing_data)
    X_train, X_test, Y_train, Y_test = model.Split(processing_labeld_data)
    # reading data
    act_fun = AC_Value.get()

    hidden = hiddenlayers_TextField.get()
    hidden_int = hiddenlayers_TextField.getint(hidden)

    no_neurons = neuorns_hiddenlayers_TextField.get()
    no_neurons_int = string_to_list(no_neurons, hidden_int, Y_train)

    le_str = learningRate_TextField.get()
    le_rate = learningRate_TextField.getdouble(le_str)

    epochs_str = number_Of_Epochs_TextField.get()
    epochs = number_Of_Epochs_TextField.getint(epochs_str)

    bias = biasCheckBox.get()
    test_on = TestCheckBox.get()
    param = model.Mlp(no_neurons_int, X_train, Y_train,
                      act_fun, epochs, le_rate, bias)
    mat, acc = model.confusionMatrix(
        param, X_test, Y_test, act_fun, bias, test_on, X_train, Y_train)

    show_acc(mat, acc, test_on)


# GUI
Activation_Functions = [
    "sigmoid",
    "tanh",
]


def show_acc(mat, acc, Test):
    top = Toplevel(MAIN)
    top.geometry("350x500")
    top.title("Values")
    if (Test == True):
        test_text_label = Label(top, text="Values of Test:")
        test_text_label.pack()
    if (Test == False):
        test_text_label = Label(top, text="Values of Train:")
        test_text_label.pack()

    # accurcy
    accurcy_text_label = Label(top, text="ACCURCY IS:")
    accurcy_text_label.pack()
    accurcy_value = str(acc)
    accurcy_value_label = Label(top, text=accurcy_value)
    accurcy_value_label.pack()
    # mat

    mat_text_label = Label(top, text="Confussion Matrix IS:")
    mat_text_label.pack()
    mat_value = str(mat)
    mat_value_label = Label(top, text=mat_value)
    mat_value_label.pack()


def open_popup(w, b, acc):
    top = Toplevel(MAIN)
    top.geometry("350x500")
    top.title("Values")
    # weight&bias
    weight_text_label = Label(top, text="WEIGHT IS:")
    weight_text_label.pack()
    weight_value = str(w)
    weight_value_label = Label(top, text=weight_value)
    weight_value_label.pack()
    bias_text_label = Label(top, text="BIAS IS:")
    bias_text_label.pack()
    bias_value = str(b)
    bias_value_label = Label(top, text=bias_value)
    bias_value_label.pack()
    # accurcy
    accurcy_text_label = Label(top, text="ACCURCY IS:")
    accurcy_text_label.pack()
    accurcy_value = str(acc)
    accurcy_value_label = Label(top, text=accurcy_value)
    accurcy_value_label.pack()


if __name__ == '__main__':

    # Main window
    MAIN = Tk()
    MAIN.title('Task NN 3')
    MAIN.geometry("350x500")

    ########################
    # Add number of hidden layers
    hiddenlayers_Header = Label(MAIN, text="Add number of hidden layers")
    hiddenlayers_Header.pack()

    hiddenlayers_TextField = ttk. Entry(MAIN, width=20)
    hiddenlayers_TextField.pack()
    ############################
    # Number of neurons in each layer seprating by [,]
    neuorns_hiddenlayers_Header = Label(
        MAIN, text="Number of neurons in each layer seprating by [,]")
    neuorns_hiddenlayers_Header.pack()

    neuorns_hiddenlayers_TextField = ttk. Entry(MAIN, width=20)
    neuorns_hiddenlayers_TextField.pack()
    ###############################

    # Select Features
    Header_function = Label(MAIN, text="Select Activation function")
    Header_function.pack()
    # 1
    AC_Value = StringVar()
    AC_Value.set("Activation Functions")
    AC_DropMenu = OptionMenu(MAIN, AC_Value, *Activation_Functions)
    AC_DropMenu.pack()

    ##########################

    # Add Learning Rate
    learningRate_Header = Label(MAIN, text="Add Learning Rate")
    learningRate_Header.pack()

    learningRate_TextField = ttk. Entry(MAIN, width=20)
    learningRate_TextField.pack()

    ###############################

    # Add Epochs
    number_Of_Epochs_Header = Label(
        MAIN, text="Add Number Of Epochs")
    number_Of_Epochs_Header.pack()
    number_Of_Epochs_TextField = Entry(MAIN, width=20)
    number_Of_Epochs_TextField.pack()

    # Select Bias
    biasCheckBox = IntVar()
    checkbox = Checkbutton(MAIN, text='Bias',
                           variable=biasCheckBox)
    checkbox.pack()
    # select test
    TestCheckBox = IntVar()
    checkbox = Checkbutton(MAIN, text='TEST',
                           variable=TestCheckBox)
    checkbox.pack()

    # Start Classification

    button = Button(MAIN, text="Start Modeling",
                    command=Data_from_gui_to_Model)
    button.pack()

    sep = Separator(MAIN, orient='horizontal')
    sep.pack(fill='x')
    button = Button(MAIN, text="Bonus",
                    command=bonus)
    button.pack()

    # Select feature to plot graph
    '''plotGraphBtn2 = Button(MAIN, text='Plot Graph Of 3 classes',
                           command=lambda: plotGraph(feature1_Value.get(), feature2_Value.get())).pack()'''

    MAIN.mainloop()
