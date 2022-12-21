"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = 4  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        """
        x_i -> Dx1
        w_iT -> 1XD
        w_i -> DX1
        """
        self.w = np.zeros((X_train.shape[1], self.n_class))
        for t in range(0, self.epochs):
            for i in range(0, X_train.shape[0]):
                y_hat = None
                y_score = None
                for c in range(0, self.n_class):
                    w_c = self.w[:,c]
                    x_i = X_train[i]
                    c_score = np.dot(w_c, x_i)
                    if y_score is None or c_score > y_score:
                        y_score = c_score
                        y_hat = c
                if y_hat != y_train[i]:
                    for c in range(0, self.n_class):
                        if c != y_train[i]:
                            self.w[:,c] -= self.lr * (1/(t+1)) * X_train[i]
                        else:
                            self.w[:,c] += self.lr * (1/(t+1)) * X_train[i]
                        

                

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
        pred = np.zeros(X_test.shape[0])
        for i in range(0, X_test.shape[0]):
            y_hat = None
            y_score = None
            for c in range(0, self.n_class):
                w_c = self.w[:,c]
                x_i = X_test[i]
                c_score = np.dot(w_c, x_i)
                if y_score is None or c_score > y_score:
                    y_score = c_score
                    y_hat = c
            pred[i] = int(y_hat)
        return pred
