import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / (X.shape[0] - 1)
        # 求协方差矩阵的特征值和特征向量
        A, B = np.linalg.eig(self.covariance)
        # 对特征值进行降序排序
        idx = np.argsort(-1 * A)
        # 降维矩阵
        self.components = B[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components)


if __name__ == '__main__':
    pca = PCA(n_components=2)
    X = np.array([[2.5, 2.4, 1.0, 3.5],
                  [0.5, 0.7, 2.0, 0.5],
                  [2.2, 2.9, 1.5, 3.0],
                  [1.9, 2.2, 1.3, 2.7],
                  [3.1, 3.0, 0.8, 4.1]])
    Xnew = pca.fit_transform(X)
    print(Xnew)
