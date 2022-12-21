"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return X * (X > 0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        c = np.max(X, axis=1)[:,np.newaxis]
        c = np.log(np.sum(np.exp(X - c), axis=1))[:,np.newaxis] + c
        return np.exp(X - c)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        layer_input = X
        for i in range(1, self.num_layers + 1):
            W_i = self.params["W{}".format(i)]
            b_i = self.params["b{}".format(i)]
            layer_raw_output = self.linear(W_i, layer_input, b_i)

            layer_relu_output = layer_raw_output
            if i != self.num_layers:
                layer_relu_output = self.relu(layer_raw_output)
            self.outputs[i] = layer_relu_output
            layer_input = layer_relu_output
            
        return self.softmax(self.outputs[self.num_layers])
    
    def adam_update(self, gradients, parameters, i, b1, b2, epsilon, epoch, lr):
        if "W{}m".format(i) not in self.gradients:
            self.gradients["W{}m".format(i)] = 0
            self.gradients["W{}v".format(i)] = 0
            self.gradients["b{}m".format(i)] = 0
            self.gradients["b{}v".format(i)] = 0
          
        for p_type in ["W", "b"]:
            m_prev = self.gradients["{}{}m".format(p_type, i)]
            v_prev = self.gradients["{}{}v".format(p_type, i)]
            param = parameters[p_type]
            grad = gradients[p_type]
            
            m_new = b1*m_prev + (1-b1)*grad
            v_new = b2*v_prev + (1-b2)*np.square(grad)
            m_hat = m_new / (1-b1**(epoch+1))
            v_hat = v_new / (1-b2**(epoch+1))
            param += -lr * m_hat / (np.sqrt(v_hat) + epsilon)

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0, b1 = None, b2 = None, epsilon = None, 
        epoch = None
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        layer_outs = [X]
        for i in range(1, self.num_layers + 1):
            layer_outs.append(self.outputs[i])
        
        probs = self.softmax(layer_outs[self.num_layers])
        num_examples = X.shape[0]
        correct_logprobs = -np.log(probs[range(num_examples),y] + 1e-999)
        
        W_k = self.params["W{}".format(self.num_layers)]
        b_k = self.params["b{}".format(self.num_layers)]
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W_k*W_k)
        loss = data_loss + reg_loss
        
        self.gradients = {}
        adam = b1 is not None and b2 is not None and epsilon is not None and epoch is not None
        
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples
        self.gradients[self.num_layers] = dscores
        
        Z = layer_outs[self.num_layers - 1]
        dW_k = np.dot(Z.T, dscores)
        dW_k += reg*W_k
        db_k = np.sum(dscores, axis=0, keepdims=True)[0]
        
        if adam:
            self.adam_update({"W":dW_k,"b":db_k}, {"W":W_k,"b":b_k}, self.num_layers, b1, b2, epsilon, epoch, lr)
        else:
            W_k += -lr * dW_k
            b_k += -lr * db_k

        if self.num_layers < 1:
            return loss
        
        upstream = dscores
        for i in range(self.num_layers - 1, 0, -1):
            W_i = self.params["W{}".format(i)]
            W_i_1 = self.params["W{}".format(i+1)]
            b_i = self.params["b{}".format(i)]
            hidden_layer = self.outputs[i]
            
            dhidden = np.dot(upstream, W_i_1.T)
            dhidden[hidden_layer <= 0] = 0
            self.gradients[i] = dhidden
            
            dW_i = np.dot(layer_outs[i-1].T, dhidden)
            dW_i += reg*W_i
            db_i = np.sum(dhidden, axis=0, keepdims=True)[0]
            
            if adam:
                self.adam_update({"W":dW_i,"b":db_i}, {"W":W_i,"b":b_i}, i, b1, b2, epsilon, epoch, lr)
            else: 
                W_i += -lr * dW_i
                b_i += -lr * db_i
            
            upstream = dhidden
        return loss
