import numpy as np
from ..utils import label_to_onehot, append_bias_term

class LogisticRegression(object):
    """
    Logistic regression classifier using softmax and cross-entropy loss for multiclass classification.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the logistic regression model.
        Arguments:
            lr (float): Learning rate of the gradient descent
            max_iters (int): Maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

    def softmax(self, data, w):
        exponential = np.exp(data @ w)
        res = exponential / np.sum(exponential, axis=1, keepdims=True)
        return res

    

    def fit(self, training_data, training_labels):
        """
        Fit the logistic regression model using gradient descent.
        Arguments:
            training_data (array): Training data of shape (N, D)
            training_labels (array): True labels of shape (N,)
        """
        one_hot_labels = label_to_onehot(training_labels)
        n_features = training_data.shape[1]
        n_classes = one_hot_labels.shape[1]
        self.weights = np.zeros(shape=(n_features, n_classes))

        for i in range(self.max_iters):
            predictions = self.softmax(training_data, self.weights)
            gradient = training_data.T @ (predictions - one_hot_labels)
            self.weights -= self.lr*gradient
        return self.predict(training_data)
            

    def predict(self, test_data):
        """
        Predict using the logistic regression model.
        Arguments:
            test_data (array): Test data of shape (N, D)
        Returns:
            pred_labels (array): Predicted labels of shape (N,)
        """
        prediction = self.softmax(test_data, self.weights)
        return np.argmax(prediction, axis=1)
