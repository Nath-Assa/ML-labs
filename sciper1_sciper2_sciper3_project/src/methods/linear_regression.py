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
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
       # Adding a column of ones to incorporate the bias term in the model
        ones = np.ones((training_data.shape[0], 1))
        X = np.hstack((ones, training_data))
        
        # Regularization matrix (lambda*I); lambda on the diagonal, excluding the bias term
        L = self.lmda * np.eye(X.shape[1])
        L[0, 0] = 0  # No regularization for the bias term
        
        # Closed-form solution (X'X + L)^(-1)X'Y
        self.theta = np.linalg.inv(X.T @ X + L) @ X.T @ training_labels
        
        pred_regression_targets = self.predict(training_data)
        
        return pred_regression_targets


def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ones = np.ones((test_data.shape[0], 1))
        X = np.hstack((ones, test_data))
        
        # Predict using the calculated theta
        pred_regression_targets = X @ self.theta

        return pred_regression_targets
