import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='closed', alpha=0.1, fit_intercept=True, regularization='', reg_intercept=False, penalty=0.1, n_iters=400):
        '''
        method:[None or 'closed', 'cd', 'gd'] - 'cd': Coordinate Descent method, None or 'closed': , gd': Gradient Descent method
        alpha:float - if method is 'gd': Learning rate in case of Gradient descent
                      if method is 'cd' and regularization is 'elastic-net': Elastic-net alpha parameter
        # TODO - LAR Method (Pg 94, Elements of Statistical Leanring)
        regularization:[None, 'l1' | 'lasso', 'l2' | 'ridge'] - None: Ordinary Least Squares method,
                                                                'l1' or 'lasso': Lasso Regression which is Least Squares with L1 regularization / penalty
                                                                'l2' or 'ridge': Ridge Regression which is Least Squares with L2 regularization / penalty
        # TODO - Elastic-Net regularization (Pg 93, Elements of Statistical Leanring)
        fit_intercept:bool - To use the intercept term, if False, X is assumed to be centered
        penalty:float - a.k.a lambda, multiplier that controls 
        n_iters:int - number of iterations in case of Gradient Descent Method
        '''
        method = method.lower()
        if method == 'gd':
            assert regularization is None, "Gradient descent cannot be used with regularization, use Coordinate Descent"
        self.method = method
        regularization = regularization.lower()
        if regularization == 'l1' or regularization == 'lasso':
            self.reg = 'l1'
        elif regularization == 'l2' or regularization == 'ridge':
            self.reg = 'l2'
        else:
            self.reg = None
        self.penalty = penalty
        self.reg_intercept = reg_intercept
        self.fit_intercept = fit_intercept
        self.W = np.array([])
        self.alpha = alpha
        self.n_iters = n_iters
    
    def train(self, X, y):
        '''
        X - (n_datapoints, n_features)
        y - (n_datapoints, 1)
        '''
        if self.fit_intercept:
            # To accomodate the bias / y-intercept term
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        n_datapoints, n_features = X.shape
        assert y.size == n_datapoints, "X and y must have same number of rows"
        W = np.random.rand(n_features, 1)
        try:
            # Closed form
            if self.method == 'closed':
                if self.reg is None:
                    W = np.linalg.pinv(X.T@X) @ X.T @ y
                    # Use pinv (Moore-Penrose Pseudo inverse) so that the case (X.T*X) is non-invertible is handled
                    # Cases when (X.T*X) might be non-invertible - 1. linearly dependent features, 2. too many features (i.e n_datapoints <= n_features)
                elif self.reg == 'l1':
                    print("Lasso Regression doesn't have a closed form")
                    raise ValueError
                elif self.reg == 'l2':
                    I = np.eye(n_features)
                    if not(self.reg_intercept):    # Don't regularize intercept term
                        I[0, 0] = 0
                    W = np.linalg.pinv(X.T@X+self.alpha*I) @ X.T @ y
                else:
                    print("Unrecognized regularization for Closed form method")
                    raise NotImplemented
            # Using Gradient Descent
            elif self.method == 'gd':
                if self.reg is None:
                    for i in range(self.n_iters):
                        W -= self.alpha / n_datapoints * (X.T @ (np.dot(X, W).ravel() - y))
                elif self.reg == 'l1':
                    # http://www.cs.cmu.edu/afs/cs/project/link-3/lafferty/www/ml-stat2/talks/YondaiKimGLasso-SLIDE-YD.pdf
                    print("Gradient Descent not implemented for Lasso")
                    raise NotImplementedError
                elif self.reg == 'l2':
                    if self.reg_intercept and self.fit_intercept:
                        for i in range(self.n_iters):
                            W[0] -= self.alpha / n_datapoints * (X[:, 0] @ (X[:, 0]*W[0] - y))
                            W[1:] = W[1:]*(1-self.alpha * self.penalty / n_datapoints) - \
                                            self.alpha / n_datapoints * (X[:, 1:].T @ (np.dot(X[:, 1:], W[1:]).ravel() - y))
                    else:
                        for i in range(self.n_iters):
                            W -= self.alpha / n_datapoints * (X.T @ (np.dot(X, W).ravel() - y))
            # Using Coordinate Descent
            elif self.method == 'cd':
                i = 0
                if self.reg is None:
                    print("Not yet implemented")
                    raise NotImplementedError
                elif self.reg == 'l1':
                    z = np.sum(X**2, axis=0)
                    while i < self.n_iters:
                        for j in range(n_features):
                            residuals = y - np.dot(X, W).ravel()
                            # Track cost
                            # cost.append(float(residuals.T@residuals) + self.penalty*np.sum(np.abs(W)))
                            rho = X[:, j] @ (residuals + W[j]*X[:, j])
                            if not(self.reg_intercept) and self.fit_intercept and j == 0:
                                W[j] = rho / z[j]
                            else:
                                # Soft-thresholding
                                if rho < -self.penalty /2:
                                    W[j] = (rho + self.penalty / 2) / z[j]
                                elif rho > self.penalty:
                                    W[j] = (rho - self.penalty / 2) / z[j]
                                else:
                                    W[j] = 0
                            # Hard thresholding (Best-subset selection drops all variables with coefficients smaller than the Mth largest)
                        i += 1
                elif self.reg == 'l2':
                    print("Not yet implemented")
                    raise NotImplementedError
                elif self.reg == 'elastic-net':
                    z = np.sum(X**2, axis=0)
                    while i < self.n_iters:
                        for j in range(n_features):
                            residuals = y - np.dot(X, W).ravel()
                            # Track cost
                            # cost.append(float(residuals.T@residuals) + self.penalty*np.sum(np.abs(W)))
                            rho = X[:, j] @ (residuals + W[j]*X[:, j])
                            if not(self.reg_intercept) and self.fit_intercept and j == 0:
                                W[j] = rho / z[j]
                            else:
                                # Soft-thresholding
                                if rho < -self.penalty /2:
                                    W[j] = (rho + self.penalty / 2) / z[j]
                                elif rho > self.penalty:
                                    W[j] = (rho - self.penalty / 2) / z[j]
                                else:
                                    W[j] = 0
                            # Hard thresholding (Best-subset selection drops all variables with coefficients smaller than the Mth largest)
                        i += 1
                    #self.penalty * np.sum(self.alpha * self.W**2 + (1-self.alpha) * np.abs(self.W))
            else:
                print("Unknown method, might not be implemented yet, training failed")
                raise NotImplementedError
            self.W = W
        except Exception as e:
            print(X.shape, self.W.shape, y.shape, n_features, n_datapoints)
            print(e.stackTrace())

    def predict(self, X):
        if self.fit_intercept:
            return (np.insert(X, 0, np.ones(X.shape[0]), axis=1)@self.W).ravel()
        return (X@self.W).ravel()
    
    @staticmethod
    def r2_score(y, y_pred):
        if y.ndim > 1:
            y = y.reshape(-1)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(-1)
        y_mean = y.mean()
        return 1 - ((y-y_pred)@(y-y_pred)) / ((y-y_mean)@(y-y_mean))

class Lasso(LinearRegression):
    '''
    Use regularization = penalty * L1
    Use coordinate dedscent by default Loss function is convex and non-differentiable
    '''
    def __init__(self, method='cd', fit_intercept=True, alpha=0.1, penalty=0.1, n_iters=400):
        super().__init__(method=method, fit_intercept=fit_intercept, alpha=alpha, regularization='l1', penalty=penalty, n_iters=n_iters)
        
class Ridge(LinearRegression):
    '''
    Use regularization = penalty * L2
    '''
    def __init__(self, method='closed', fit_intercept=True, alpha=0.1, penalty=0.1, n_iters=400):
        super().__init__(method=method, fit_intercept=fit_intercept, alpha=alpha, regularization='l2', penalty=penalty, n_iters=n_iters)
        
class ElasticNet(LinearRegression):
    '''
    Use regularization = alpha * L1 + penalty * L2
    '''
    def __init__(self, method='closed', fit_intercept=True, alpha=0.1, penalty=0.1, n_iters=400):
        super().__init__(method=method, fit_intercept=fit_intercept, alpha=alpha, regularization='elastic-net', penalty=penalty, n_iters=n_iters)    
