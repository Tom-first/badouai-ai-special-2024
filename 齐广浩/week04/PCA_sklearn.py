import numpy as np
from sklearn.decomposition import PCA

X = np.array([[2.5, 2.4, 1.0, 3.5],
              [0.5, 0.7, 2.0, 0.5],
              [2.2, 2.9, 1.5, 3.0],
              [1.9, 2.2, 1.3, 2.7],
              [3.1, 3.0, 0.8, 4.1]])
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 执行
Xnew = pca.fit_transform(X)  # 降维后的数据
print(Xnew)  # 打印数据
