# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
# 导入评价包
from sklearn import metrics
from matplotlib.colors import ListedColormap
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


# load data
iris = load_iris()
X, y = iris.data[:, :2], iris.target

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc=2) # 把说明放在左上角，具体请参考官方文档
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
# 对原特征数据进行标准化预处理, 这个其实挺重要，但是经常被一些选手忽略
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# 训练感知机模型
from sklearn.linear_model import Perceptron
# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

# 训练logistics分类
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)
lr.predict_proba(X_test[0,:])
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# # 选择使用SGD分类器，适合大规模数据，随机梯度下降方法估计参数
# clf = SGDClassifier()
# clf.fit(X_train, y_train)
# y_train_predict = clf.predict(X_train)
# # 内测，使用训练样本进行准确性能评估
# print metrics.accuracy_score(y_train, y_train_predict)
# # 标准外测，使用测试样本进行准确性能评估
# y_predict = clf.predict(X_test)
# print metrics.accuracy_score(y_test, y_predict)
#
# # 如果需要更加详细的性能报告，比如precision, recall, accuracy，可以使用如下的函数。
# print metrics.classification_report(y_test, y_predict, target_names = iris.target_names)
#
# # 如果想详细探查SGDClassifier的分类性能，我们需要充分利用数据，因此需要把数据切分为N个部分，每个部分都用于测试一次模型性能。
# from sklearn.cross_validation import cross_val_score, KFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# # 这里使用Pipeline，便于精简模型搭建，一般而言，模型在fit之前，对数据需要feature_extraction, preprocessing, 等必要步骤。
# # 这里我们使用默认的参数配置
# clf = Pipeline([('scaler', StandardScaler()), ('sgd_classifier', SGDClassifier())])
# # 5折交叉验证整个数据集合
# cv = KFold(X.shape[0], 5, shuffle=True, random_state = 33)
# scores = cross_val_score(clf, X, y, cv=cv)
# print scores
# # 计算一下模型综合性能，平均精度和标准差
# print scores.mean(), scores.std()
# from scipy.stats import sem
# import numpy as np
# print np.mean(scores), sem(scores)
