import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object.
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the Linear Regression model with a lambda parameter for ridge regression.
        """
        self.lmda = lmda
        self.theta = None  # This will store the model coefficients after fitting

    def fit(self, training_data, training_labels):
        """
            Trains the model using provided training data and labels, and returns predictions on that training data.
            Arguments:
                training_data (np.array): Training data of shape (N,D)
                training_labels (np.array): Regression target of shape (N, regression_target_size)
            Returns:
                pred_labels (np.array): Predicted labels of shape (N, regression_target_size)
        """
        # Regularization matrix setup
        L = self.lmda * np.eye(training_data.shape[1])
        if self.lmda != 0:
            L[0, 0] = 0  # Assume first column is the bias term, do not regularize it if lambda is not zero

        # Solve the normal equation with regularization
        self.theta = np.linalg.inv(training_data.T @ training_data + L) @ training_data.T @ training_labels

        # Predict on the training data using the calculated coefficients
        pred_labels = training_data @ self.theta
        return pred_labels

    def predict(self, test_data):
        """
            Predicts output using the linear model coefficients.
            Arguments:
                test_data (np.array): Test data of shape (N, D)
            Returns:
                test_labels (np.array): Predicted labels of shape (N, regression_target_size)
        """
        if self.theta is None:
            raise ValueError("Model has not been trained yet - please run fit() first.")
        return test_data @ self.theta
