
# Backpropagation Algorithm Pseudocode
```
Initialize the weights randomly
Set the learning rate and maximum number of epochs
For each epoch from 1 to max_epochs:
    For each training example (input x, true output y):
        Perform a forward pass through the network to calculate the predicted output: y_pred = f(w1*x1 + w2*x2 + ... + wn*xn)
        Calculate the error: error = y - y_pred
        For each weight in the network:
            Calculate the gradient of the loss with respect to the weight: dL/dw = -error * f'(w1*x1 + w2*x2 + ... + wn*xn) * xi
            Update the weight: w(new) = w(old) - learning_rate * dL/dw
Return the final weights
```
# Data
[***To download data of Bonus***](https://www.kaggle.com/oddrationale/mnist-in-csv)

