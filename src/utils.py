import numpy as np

def computeMean(X, Y, label):
    data = X[Y == label]
    mean = np.mean(data, axis=0)
    return np.array(mean)

def computeCovariance(X):
    data = X
    covariance = np.cov(data,rowvar=False,bias=True)
    epsilon = 1e-9
    covariance += np.eye(covariance.shape[0]) * epsilon
    return covariance

def computeAccuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
