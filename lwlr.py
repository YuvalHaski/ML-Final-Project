from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import sys


class LWLR(BaseEstimator, RegressorMixin):
    """
    A demo of Local Weight Linear Regression, 
    refer to https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/
    """
    def __init__(self,k=1, eps=1e-10):
        """
        The hyper parameter k controls the weighting effect for samples
        """
        self.k = k
        self.eps = eps
        self.X = None
        self.y = None
    
    def fit(self,X,y):
        """
        Non-parametric method, memorize training samples X and targets y
        """
        self.X = np.array(X)
        self.y = np.array(y)
        return self
        
    def predict(self, X):
        """
        Given feature vectors X, make prediction
        """
        result = []
        X = np.array(X)
        for example in X:
            prediction = self._predict_single(example)
            result.append(prediction)
            
        return np.array(result).flatten()
        
    def _predict_single(self, X_in):
        """
        Train and predict for a sinlge input feature vector X_in
        """
        X_in = np.array(X_in, dtype=float)

        # compute weights with numerical stability
        diff = np.square(self.X - X_in)
        scaled_diff = -np.sum(diff, axis=1) / (2 * self.k)
        clipped_diff = np.clip(scaled_diff, -500, 500)  # Clipping to a reasonable range
        weights = np.exp(clipped_diff + self.eps)
        
        # fit linear regression model on weighted samples
        lr = LinearRegression().fit(self.X, self.y, sample_weight=weights)
        return lr.predict(X_in.reshape(1, -1))[0]

    
    def get_params(self, deep=True):
        return {"k": self.k, "eps": self.eps}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
def search_best_k(X, y, n_folds=10, k_range=[1], scoring="neg_mean_squared_log_error"):
    """
    Search best k given a list of candidate ks
    n_folds: the number of CV split
    k_range: list of candidate ks
    scoring: sklearn acceptable scoring object or strings for model selection
    Return: tuple of (the best k, cv history)
    """
    np.random.seed(999)
    results = []
    best_k, best_score = 0, -sys.maxsize
    for k in k_range:
        scores = cross_validate(LWLR(k), X, y, cv=10, scoring=scoring, n_jobs=-1, return_train_score=True)
        avg_train_score = np.mean(scores['train_score'])
        avg_test_score = np.mean(scores["test_score"])
        results.append({"k":k, "avg_train_score":avg_train_score, "avg_test_score":avg_test_score})
        if best_score < avg_test_score:
            best_k, best_score = k, avg_test_score
        print(results[-1])
    return best_k, results
