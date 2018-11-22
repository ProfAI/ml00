import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
features = ['sepal length','sepal width','petal length','petal width','target']

iris = pd.read_csv(url, names=features)

iris.head()

X = iris.drop("target", axis=1).values
y = iris["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pc_train = pca.fit_transform(X_train)
pc_test = pca.transform(X_test)

plt.figure(figsize=(12,10))
plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.scatter(pc_train[:,0], pc_train[:,1], c=y_train)
plt.scatter(pc_test[:,0], pc_test[:,1], c=y_test, alpha=0.5)


lr = LogisticRegression()
lr.fit(pc_train, y_train)


accuracy_score(y_train, lr.predict(pc_train))
accuracy_score(y_test, lr.predict(pc_test))

log_loss(y_train, lr.predict_proba(pc_train))
log_loss(y_test, lr.predict_proba(pc_test))


np.unique(y, return_counts=True)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

ld_train = lda.fit_transform(X_train, y_train)
ld_test = lda.transform(X_test)

plt.figure(figsize=(12,10))
plt.xlabel("Primo discriminante")
plt.ylabel("Secondo discriminante")
plt.scatter(ld_train[:,0], ld_train[:,1], c=y_train)
plt.scatter(ld_test[:,0], ld_test[:,1], c=y_test, alpha=0.5)


lr = LogisticRegression()
lr.fit(ld_train, y_train)


accuracy_score(y_train, lr.predict(ld_train))
accuracy_score(y_test, lr.predict(ld_test))

log_loss(y_train, lr.predict_proba(ld_train))
log_loss(y_test, lr.predict_proba(ld_test))
