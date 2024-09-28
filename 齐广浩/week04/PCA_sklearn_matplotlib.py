import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 4个特征的数据
y = iris.target  # 标签
print("特征数据：\n", X)
print("标签：\n", y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制降维后的数据
plt.figure(figsize=(8, 6))

# 按不同的标签绘制数据点
for label, marker, color in zip([0, 1, 2], ['o', '^', 's'], ['r', 'g', 'b']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], marker=marker, color=color, label=iris.target_names[label], alpha=0.7)

# 添加图例和标签
plt.legend(loc='best')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')

# 显示图形
plt.show()
