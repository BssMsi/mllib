import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iters=400):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.W = 0

    @staticmethod
    def normalize(X):
        '''
        returns a normalized version of X where mean value of each feature is 0 and each value is in the interval [0,1]
        '''        
        return (X-X.mean(axis=0)) / (X.max(axis=0)-X.min(axis=0))
    
    def train(self, X, y):
        '''
        X - (n_datapoints, n_features)
        y - (n_datapoints, 1)
        '''
        
        n_datapoints, n_features = X.shape
        assert y.size == n_datapoints
        # Normalize
        if self.normalize:
            X = self.normalize(X)
        # To accomodate the bias / y-intercept term
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        # Gradient Descent
#         W = np.random.rand(n_features, 1)
#         for i in range(self.n_iters):
#             W -= self.learning_rate / n_datapoints * (X.T * (np.dot(X, W).ravel() - y))
        # Use pinv (Moore-Penrose Pseudo inverse) so that the case (X.T*X) is non-invertible is handled
        # Cases when (X.T*X) might be non-invertible - 1. linearly dependent features, 2. too many features (i.e n_datapoints <= n_features)
        self.W = np.linalg.pinv(X.T@X) @ X.T @ y
        
    def predict(self, X):
        return np.insert(self.normalize(X), 0, np.ones(X.shape[0]), axis=1)@self.W
    
    def plot(self, X, y, feature=0):
        plt.scatter(X[:, feature], y)