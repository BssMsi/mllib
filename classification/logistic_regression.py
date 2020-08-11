import numpy as np


class LogisticRegression:
    '''
    method:['gd', 'cd', 'bfgs', 'lbfgs', 'liblinear'] - https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-defintions
    https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
    alpha:float - if method is 'gd': Learning rate in case of Gradient descent
                  if method is 'cd' and regularization is 'elastic-net': Elastic-net alpha parameter
    regularization:[None, 'l1', 'l2'] - None: No regularization,
                                        'l1': L1 penalty = absolute sum of the weights
                                        'l2': L2 penalty = sum of squares of the weights
    fit_intercept:bool - To use the intercept term, if False, X is assumed to be centered
    penalty:float - a.k.a lambda, multiplier that controls 
    n_iters:int - number of iterations in case of Gradient Descent Method
    '''
    def __init__(self, method='gd', alpha=0.1, fit_intercept=False, regularization='l2', reg_intercept=False, penalty=1.0, n_iters=400):
        if regularization is None:
            self.reg = None
        else:
            regularization = regularization.lower()
            if regularization == 'l1' or regularization == 'lasso':
                self.reg = 'l1'
            elif regularization == 'l2' or regularization == 'ridge':
                self.reg = 'l2'
        method = method.lower()
        if method == 'gd':
            assert self.reg == 'l2' or self.reg is None, "Gradient descent method can only be used with L2 regularization or None"
        self.method = method
        self.penalty = penalty
        self.reg_intercept = reg_intercept
        self.fit_intercept = fit_intercept
        self.W = np.array([])
        self.n_iters = n_iters
        self.alpha = alpha

    @staticmethod
    def sigmoid(z):
        return 1 / (1+np.exp(-z))
        
    def train(self, X, y, return_history=False):
        '''
        X - (n_datapoints, n_features)
        y - (n_datapoints, 1)
        '''
        if self.fit_intercept:
            # To accomodate the bias / y-intercept term
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        n_datapoints, n_features = X.shape
        assert y.size == n_datapoints, "X and y must have same number of rows"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_classes = len(np.unique(y))
        if n_classes > 2:
            self.mode = 'ovr'
            W = np.zeros((n_features, n_classes))
            cost = []
            # One vs rest strategy
            for c in range(n_classes):
                y_c = np.where(y == c, 1, 0)
                if return_history:
                    W_c, cost_c = self._algo(X, y_c, True)
                    W[:, c] = W_c.ravel()
                    cost.append(cost_c)
                else:
                    W[:, c] = self._algo(X, y_c, False).ravel()
        else:
            self.mode = 'binary'
            if return_history:
                W, cost = self._algo(X, y, True)
            else:
                W = self._algo(X, y, False)
        self.W = W
        if return_history:
            return cost
        return self

    def _algo(self, X, y, cost_history):
        n_datapoints, n_features = X.shape
        W = np.zeros((n_features, 1))
        cost = []
        # Using Gradient Descent
        if self.method == 'gd':
            if self.reg is None:
                if cost_history:
                    for i in range(self.n_iters):
                        h = self.sigmoid(X@W)
                        W -= self.alpha * (X.T @ (h - y)) / n_datapoints
                        cost.append((- y.T@np.log(h) - (1-y.T)@np.log(1-h)).item() / n_datapoints)
                else:  
                    for i in range(self.n_iters):
                        W -= self.alpha * (X.T @ (self.sigmoid(X@W) - y)) / n_datapoints
            elif self.reg == 'l1':
                print("Gradient Descent not implemented for L1")
                raise NotImplementedError
            elif self.reg == 'l2':
                if self.fit_intercept and not(self.reg_intercept):
                    if cost_history:
                        for i in range(self.n_iters):
                            h = self.sigmoid(X@W)
                            W[0] = W[0] - self.alpha * (X[:, 0] @ (h - y)) / n_datapoints
                            W[1:] = W[1:] - self.alpha * (X[:, 1:].T @ (h - y) + self.penalty * W[1:].sum()) / n_datapoints
                            cost.append((- y.T@np.log(h) - (1-y.T)@np.log(1-h) + self.penalty * (W**2).sum() / 2 ).item() / n_datapoints)
                    else:
                        for i in range(self.n_iters):
                            h = self.sigmoid(X@W)
                            W[0] = W[0] - self.alpha * (X[:, 0] @ (h - y)) / n_datapoints
                            W[1:] = W[1:] - self.alpha * (X[:, 1:].T @ (h - y) + self.penalty * W[1:].sum()) / n_datapoints
                else:
                    if cost_history:
                        for i in range(self.n_iters):
                            W = W - self.alpha * (X.T @ (self.sigmoid(X@W) - y) + self.penalty * W.sum()) / n_datapoints
                            cost.append((- y.T@np.log(h) - (1-y.T)@np.log(1-h) + self.penalty * (W**2).sum() / 2 ).item() / n_datapoints)
                    else:
                        for i in range(self.n_iters):
                            W = W - self.alpha * (X.T @ (self.sigmoid(X@W) - y) + self.penalty * W.sum()) / n_datapoints
        else:
            print("Unknown method, might not be implemented yet, training failed")
            raise NotImplementedError
        if cost_history:
            return W, cost
        return W
        
    def predict(self, X, return_prob=False, threshold=0.5):
        if self.fit_intercept:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        prob = self.sigmoid(X@self.W)
        if self.mode == 'ovr':
            pred = prob.argmax(axis=1).ravel()
        elif self.mode == 'binary':
            pred = np.where(prob > threshold, 1, 0).ravel()
        else:
            raise ValueError
        if return_prob:
            return pred, np.around(prob, decimals=2)
        return pred

    @staticmethod
    def score(y, y_pred):
        return 100 * (y==y_pred).sum() / len(y)