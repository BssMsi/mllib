import numpy as np


class KNN:
    def __init__(self, K=5, weights='uniform', distance='minkowski', p=2, epsilon=1e-6):
        '''
        weights:['uniform', 'distance'] - uniform: All points in each neighborhood are weighted equally.
                                          distance: weight points by the inverse of their distance
                                                  (closer neighbors of a query point will have a greater influence than neighbors which are further away)
        distance:['manhattan','taxicab','euclidean','chebyshev', minkowski','lorentzian','canberra','cosine'] - 
        https://arxiv.org/pdf/1708.04321.pdf#:~:text=Euclidean%20distance%20is%20the%20most,number%20of%20datasets%2C%20or%20both
                  Minkowski = sum(abs(x1-x2)^p+abs(y1-y2)^p)
                  Euclidean = Minkowski Distance(p=2)
                  Manhattan/Taxicab = Minkowski Distance(p=1)
                  Chebyshev = sum(abs(x1-x2)^p,abs(y1-y2)^p)
                  Lorentzian = sum(ln(1+abs(x1-x2)))
                  Canberra = sum(abs(x1-x2) / (abs(x1)+abs(x2)))
                  TODO - Hamming, mahalanobis, braycurtis, etc
        epsilon:float - used only when weights='distance'. It is added to the distances only when distance is zero to avoid divide by zero warning
        TODO - Implement more efficient storage methods like KD Tree and Ball Tree
        '''
        assert isinstance(K, int), "K must be an integer"
        self.K = K
        if weights is None or weights.lower() == 'uniform':
            self.weights = 'uniform'
        elif weights.lower() == 'distance':
            self.weights = 'distance'
        else:
            raise ValueError("Unrecognized value for weights, must be 'uniform' or 'distance'")
        distance = distance.lower()
        assert distance in ['manhattan', 'taxicab', 'euclidean', 'chebyshev', 'minkowski','lorentzian','canberra','cosine'], \
                f"Unrecognized distance '{distance}'"
        if distance == 'minkowski':
            if isinstance(p, float):
                assert np.isinf(p), "float type for p can be +-numpy.inf only"
            else:
                assert isinstance(p, int) and p >= 1, "p must be an integer and >= 1"
            self.p = p
        elif distance == 'euclidean':
            distance = 'minkowski'
            self.p = 2
        elif distance == 'manhattan' or distance == 'taxicab':
            distance = 'minkowski'
            self.p = 1
        elif distance == 'chebyshev':
            distance = 'minkowski'
            self.p = np.inf
        self.distance = distance
        self.epsilon = epsilon

    def train(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X_test, return_distances=False):
        '''
        top_distances:(K, n_datapoints)
        '''
        X_train = self.X
        n_datapoints, n_features = X_test.shape
        assert n_features == X_train.shape[1], "Training and Prediction data must have same number of features"
        # Broadcast matrices to 3 dimensions for vectorized computation
        broadcasted_X1 = np.tile(np.expand_dims(X_train, axis=2), (1,1,n_datapoints))
        broadcasted_X2 = np.tile(np.expand_dims(X_test, axis=2).T, (X_train.shape[0],1,1))
        # Caclulate feature differences, 1st dimension corresponds to the training points and 3rd dimension corresponds to the test data
        diff = np.abs(broadcasted_X1-broadcasted_X2)
        # Calculate distance
        if self.distance == 'minkowski':
            if self.p == np.inf:
                all_distances = diff.max(axis=1)
            elif self.p == -np.inf:
                all_distances = diff.min(axis=1)
            else:
                all_distances = np.power((diff**self.p).sum(axis=1), 1./self.p)
        elif self.distance == 'lorentzian':
            all_distances = np.log(1+diff).sum(axis=1)
        elif self.distance == 'canberra':
            all_distances = (diff / (broadcasted_X1+broadcasted_X2)).sum(axis=1)
        elif self.distance == 'cosine':
            all_distances = 1 - (broadcasted_X1*broadcasted_X2).sum(axis=1) / \
                            (np.sqrt((broadcasted_X1**2).sum(axis=1)) * np.sqrt((broadcasted_X2**2).sum(axis=1)))
        else:
            raise NotImplementedError        
        # Calculate Prediction
        yp = np.full((n_datapoints,), -1)        
        indices = np.argsort(all_distances, axis=0)[:self.K]
        dummy_i = np.repeat(np.arange(indices.shape[1]).reshape(1, -1), indices.shape[0], axis=0)
        top_distances = all_distances[indices, dummy_i]
        classes = self.y[indices]
        if self.weights == 'uniform':
            # Majority
            for i in range(n_datapoints):
                top_nn, counts = np.unique(classes[:, i], return_counts=True)
                yp[i] = top_nn[np.argsort(counts)[-1]]
        elif self.weights == 'distance':
            top_distances += self.epsilon
            # Inverse distance Weighted Majority
            for i in range(n_datapoints):
                w_d = -1
                for c in np.unique(self.y[indices[:,i]]):
                    #temp = (1./(all_distances[indices[:,i],i][self.y[indices[:,i]] == c]+self.epsilon)).sum()
                    temp = (1./(top_distances[classes[:,i] == c, i])).sum()                    
                    if temp > w_d:
                        w_d = temp
                        yp[i] = c
        if return_distances:
            return yp, top_distances
        return yp

    @staticmethod
    def score(y, y_pred):
        return 100 * (y==y_pred).sum() / len(y)