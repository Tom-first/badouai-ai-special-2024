"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""
import numpy as np


class CPCA(object):
    """
    note：保证输入的样本矩阵X shape = (m, n), m行样例，n个特征
    """

    # 初始化
    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维Z矩阵

        self.centrX = self._centralized()
        self.C = self._COV()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        print("样本矩阵X：\n", self.X)
        centrX = []
        # 求样本矩阵X特征均值
        mean = np.array([np.mean(attrx) for attrx in self.X.T])
        print("样本矩阵X的特征均值:\n", mean)
        centrX = self.X - mean  # 样本集中心化
        print("中心化后的样本集centrX：\n", centrX)

        return centrX

    def _COV(self):
        ns = np.shape(self.centrX)[0]  # 求centrX的样例数量
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 求样本矩阵的协方差矩阵C
        """
            K = np.shape(X)[0] - 1 中的设置，
            如果降维矩阵的维度K大于特征数量，
            会导致降维矩阵 U 生成错误，可以确保K的值合理，
            比如设置为 K = 2 或特征数以下
        """
        print("样本矩阵中心化的协方差矩阵C：\n", C)

        return C

    def _U(self):
        # 协方差矩阵C的特征值和特征向量
        A, B = np.linalg.eig(self.C)  # A是协方差矩阵C的特征值 B是C的特征向量
        print("协方差矩阵C的特征值：\n", A)
        print("协方差矩阵C的特征向量：\n", B)
        # 对协方差矩阵C的特征值进行降序排序
        ind = np.argsort(-1 * A)  # argsort正常是升序排序，这里的-1表示降序排序
        # 构建K阶降维转换矩阵U
        UT = [B[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("协方差矩阵C的转换矩阵U：\n", U)

        return U

    def _Z(self):
        """
        Z = XU
        :return: Z
        """
        Z = np.dot(self.X, self.U)
        print("X shape:\n", np.shape(self.X))
        print("U shape:\n", np.shape(self.U))

        print("Z shape:\n", np.shape(Z))
        print("样本矩阵X的降维矩阵Z：\n", Z)

        return Z


if __name__ == '__main__':
    # X = np.array([[2.5, 2.4, 1.0, 3.5],
    #                  [0.5, 0.7, 2.0, 0.5],
    #                  [2.2, 2.9, 1.5, 3.0],
    #                  [1.9, 2.2, 1.3, 2.7],
    #                  [3.1, 3.0, 0.8, 4.1]])
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print("5行4列，5个样本，每个样本4个特征:\n", X)
    pca = CPCA(X, K)
