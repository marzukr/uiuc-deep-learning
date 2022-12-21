"""Support Vector Machine (SVM) model."""

import numpy as np
import math

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        gradient = np.zeros_like(self.w)
        n = X_train.shape[0]
        for i in range(0, n):
            x_i = X_train[i]
            y_i = y_train[i]
            for c in range(0, self.n_class):
                w_c = self.w[:,c]
                gradient[:,c] += self.reg_const/n * w_c
                if c == y_i:
                    for c_c in range(0, self.n_class):
                        if c_c != y_i:
                            indicator = np.dot(self.w[:,y_i], x_i)
                            indicator -= np.dot(self.w[:,c_c], x_i)
                            if indicator < 1:
                                indicator = 1
                            else:
                                indicator = 0
                            gradient[:,c] -= indicator * x_i
                else:
                    indicator = np.dot(self.w[:,y_i], x_i)
                    indicator -= np.dot(w_c, x_i)
                    if indicator < 1:
                        indicator = 1
                    else:
                        indicator = 0
                    gradient[:,c] += indicator * x_i
        return gradient

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.zeros((X_train.shape[1], self.n_class))
        p = np.random.permutation(len(X_train))
        X_shuffle = X_train[p]
        Y_shuffle = y_train[p]
        n = X_shuffle.shape[0]
        batches = 8
        batch_size = math.ceil(n/16)
        for t in range(0, self.epochs):
            batch_start = 0
            for b in range(0, 16):
                batch_end = min(batch_start + batch_size, n)
                mini_batch_x = X_shuffle[batch_start:batch_end]
                mini_batch_y = Y_shuffle[batch_start:batch_end]
                self.w -= self.alpha/(t+1) * self.calc_gradient(mini_batch_x, mini_batch_y)
                # for i in range(0, mini_batch_x.shape[0]):
                #     x_i = mini_batch_x[i]
                #     y_i = mini_batch_y[i]
                #     for c in range(0, self.n_class):
                #         self.w[:,c] *= (1-self.alpha*self.reg_const/n)
                #         margin = np.dot(self.w[:,y_i], x_i) - np.dot(self.w[:,c], x_i)
                #         if c != y_i and margin < 1:
                #             self.w[:,y_i] += self.alpha/(t+1) * x_i
                #             self.w[:,c] -= self.alpha/(t+1) * x_i
                batch_start = batch_end
            if batch_start >= n:
                continue

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        """
        w = DxC
        x = NxD
        xwT = NxC
        """
        xwT = np.dot(X_test, self.w)
        predict = np.argmax(xwT, axis=1)
        return predict

