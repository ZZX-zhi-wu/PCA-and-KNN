# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
# 解决后端兼容性问题：强制设置稳定的后端（TkAgg/QtAgg/Agg 三选一，优先TkAgg）
plt.switch_backend('TkAgg')  # 关键：更换后端，避免PyCharm内置后端报错
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 解决中文显示问题：只保留Windows默认的"SimHei"字体，避免警告
try:
    plt.rcParams["font.family"] = ["SimHei"]  # Windows系统通用中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
except:
    # 若系统无SimHei，直接用默认字体（显示英文，无警告）
    pass


### （1）数据加载与预处理（不变）
iris = load_iris()
X = iris.data  # 4维特征
y = iris.target  # 3类标签
feature_names = iris.feature_names
target_names = iris.target_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


### （2）PCA降维与方差分析（不变，仅修改绘图显示逻辑）
# 分析所有主成分的方差贡献率
pca_all = PCA(n_components=4)
X_pca_all = pca_all.fit_transform(X_scaled)
var_ratio = pca_all.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

# 绘制方差贡献率图（用plt.savefig()替代plt.show()，避免后端问题；也可保留plt.show()）
plt.figure(figsize=(12, 5))

# 子图1：各主成分方差贡献率
plt.subplot(1, 2, 1)
plt.bar(range(1, 5), var_ratio, color='skyblue')
plt.xlabel('主成分序号')
plt.ylabel('方差贡献率')
plt.title('各主成分的方差贡献率')
plt.xticks(range(1, 5))

# 子图2：累计方差贡献率
plt.subplot(1, 2, 2)
plt.plot(range(1, 5), cum_var, 'o-', color='orange')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
plt.xlabel('主成分数量')
plt.ylabel('累计方差贡献率')
plt.title('累计方差贡献率曲线')
plt.xticks(range(1, 5))
plt.legend()

# 关键：先保存图像到本地（确保能看到图），再尝试显示（若后端兼容则显示，不兼容也已保存）
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=100)  # 保存到代码所在文件夹
plt.show()  # 若仍报错，可注释掉这行，直接看保存的图片


# 降维到2D和3D（不变）
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)


### （3）降维数据可视化（同样用“保存+显示”的逻辑）
# 1. 2D可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis',
                     edgecolor='k', s=50)
plt.xlabel(f'主成分1（贡献率：{var_ratio[0]:.2%}）')
plt.ylabel(f'主成分2（贡献率：{var_ratio[1]:.2%}）')
plt.title('PCA降维到2D的Iris数据集')
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
plt.grid(alpha=0.3)
plt.savefig('pca_2d.png', dpi=100)  # 保存图像
plt.show()

# 2. 3D可视化（可选）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=y, cmap='viridis', edgecolor='k', s=50)
ax.set_xlabel(f'主成分1（{var_ratio[0]:.2%}）')
ax.set_ylabel(f'主成分2（{var_ratio[1]:.2%}）')
ax.set_zlabel(f'主成分3（{var_ratio[2]:.2%}）')
ax.set_title('PCA降维到3D的Iris数据集')
ax.legend(handles=scatter_3d.legend_elements()[0], labels=list(target_names))
plt.savefig('pca_3d.png', dpi=100)  # 保存图像
plt.show()


### （4）KNN分类对比（不变）
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化（注意：用训练集参数转换测试集，避免数据泄露）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA降维（用训练集拟合PCA）
pca_2d = PCA(n_components=2)
X_train_pca = pca_2d.fit_transform(X_train_scaled)
X_test_pca = pca_2d.transform(X_test_scaled)

# 1. 原始高维数据KNN分类
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X_train_scaled, y_train)
y_pred_original = knn_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)

# 2. 降维后数据KNN分类
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

# 输出结果
print(f"原始高维数据（4D）的KNN分类准确率：{acc_original:.4f}")
print(f"PCA降维后（2D）数据的KNN分类准确率：{acc_pca:.4f}")