import numpy as np

def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


def calEuclidDistanceMatrix_vector(X):

    s = X.dot(X.T)

    u = (X ** 2).sum(axis = 1)
    u = u + u.reshape(-1,1)
    return (u - 2 * s)

