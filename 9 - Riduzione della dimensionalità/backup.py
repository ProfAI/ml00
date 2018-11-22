import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss


X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)
plt.scatter(X[:,0], X[:,1], c=y)


from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel='rbf', gamma=5)
kpc = kpca.fit_transform(X)

plt.scatter(kpc[:,0], kpc[:,1], c=y)

plt.scatter(kpc[:,0], np.zeros((1000,1)), c=y)

fpc = kpc[:,0]
fpc = fpc.reshape(-1,1)
fpc.shape


X_train, X_test, y_train, y_test = train_test_split(fpc, y, test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)

accuracy_score(y_train, lr.predict(X_train))
accuracy_score(y_test, lr.predict(X_test))

from scripts.viz import plot_boundary

plot_boundary(lr, X, y)
