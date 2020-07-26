import numpy as np


class NaiveBayes:
    def __init__(self, algo='gaussian', alpha=None):
        '''
        algo:str - 'gaussian': For normal distribution, Used for classification.
                   'multinomial': For multinomially distributed data
                   'bernoulli': used when feature vectors are binary
                   'complement': Adaptation of the standard Multinomial Naive Bayes (MNB) algorithm 
                       that is particularly suited for imbalanced data sets wherein the algorithm 
                       uses statistics from the complement of each class to compute the model’s weight.
        alpha:float - for 'gaussian' algorithm, alpha is the 
                      for 'multinomial' algorithm, if alpha=1, Laplace smoothing
                                        elif alpha < 1, Lidstone smoothing
                                        else alpha >= 0, prevents zero probabilities
        '''
        if algo == 'gaussian':
            self.algo = algo
            if alpha is None:
                alpha = 1e-9
            self.alpha = alpha
        elif algo == 'bernoulli':
            self.algo = algo
            if alpha is None:
                alpha = 1.0
            self.alpha = alpha
        elif algo == 'multinomial':
            self.algo = algo
            if alpha is None:
                alpha = 1.0
            self.alpha = alpha
        elif algo == 'complement':
            self.algo = algo
            if alpha is None:
                alpha = 1.0
            self.alpha = alpha            
        else:
            raise NotImplementedError

    def train(self, X, y):
        '''
        X - (n_datapoints, n_features)
        y - (n_datapoints, 1)
        '''
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_datapoints, n_features = X.shape
        assert len(y) == n_datapoints
        y = y.reshape(-1)
        
        classes = np.unique(y)
        n_classes = len(classes)
        priors = np.zeros((n_classes, 1))
        
        if self.algo == 'gaussian':
            mu = np.zeros((n_classes, n_features))
            sigma2 = np.zeros((n_classes, n_features))
            for i in range(n_classes):
                features_by_class = X[y==classes[i], :]
                priors[i] = len(features_by_class) / n_datapoints
                mu[i] = features_by_class.mean(axis=0)
                sigma2[i] = features_by_class.var(axis=0)
            self.mu = mu
            self.sigma2 = sigma2 + self.alpha * np.var(X, axis=0).max()    # To avoid numerical error (from sklearn source)
        elif self.algo == 'multinomial' or self.algo == 'bernoulli':
            likelihood = np.zeros((n_classes, n_features))
            for i in range(n_classes):
                features_by_class = X[y==classes[i], :]
                priors[i] = len(features_by_class) / n_datapoints
                temp = features_by_class.sum(axis=0)+self.alpha
                likelihood[i, :] = temp / temp.sum()
            self.likelihood = likelihood
        elif self.algo == 'complement':
            theta = np.zeros((n_classes, n_features))
            for i in range(n_classes):
                temp = self.alpha + X[y!=classes[i], :].sum(axis=0)
                theta[i, :] = temp / temp.sum()
            w = np.log(theta)
            w = w / np.abs(w).sum()
            self.w = w
        self.n_classes = n_classes
        self.priors = priors
    
    def predict(self, X, return_probs=False):
        if X.ndim < 2:
            X = X.reshape(-1, 1)
        if self.algo == 'gaussian':
            cond_prob = np.zeros((self.n_classes, X.shape[0]))
            for i in range(self.n_classes):
                cond_prob[i, :] = (np.exp(-((X-self.mu[i])**2 / (2 * self.sigma2[i]))) / np.sqrt(2*np.pi*self.sigma2[i])).prod(axis=1)
            probs = self.priors * cond_prob
            yp = probs.argmax(axis=0)
        elif self.algo == 'bernoulli':
            cond_prob = np.zeros((self.n_classes, X.shape[0]))
            for i in range(self.n_classes):
                cond_prob[i, :] = (np.power(self.likelihood[i], X) * np.power(1-self.likelihood[i], 1-X)).prod(axis=1)
            probs = self.priors * cond_prob
            yp = probs.argmax(axis=0)
        elif self.algo == 'multinomial':
            cond_prob = np.zeros((self.n_classes, X.shape[0]))
            for i in range(self.n_classes):
                cond_prob[i, :] = X @ np.log(self.likelihood[i])
            probs = np.log(self.priors) + cond_prob
            yp = probs.argmax(axis=0)
        elif self.algo == 'complement':
            probs = self.w @ X.T
            yp = probs.argmin(axis=0)
        else:
            raise NotImplementedError
        if return_probs:
            return yp, probs
        return yp
    
    @staticmethod
    def score(y, y_pred):
        return 100 * (y==y_pred).sum() / len(y)


class GaussianNB(NaiveBayes):
    def __init__(self, ):
        super().__init__(algo='gaussian')
class BernoulliNB(NaiveBayes):
    def __init__(self, ):
        super().__init__(algo='bernoulli')
class MultinomialNB(NaiveBayes):
    def __init__(self, ):
        super().__init__(algo='multinomial')
class ComplementNB(NaiveBayes):
    def __init__(self, ):
        super().__init__(algo='complement')
