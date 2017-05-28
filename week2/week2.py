import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import *

data = pd.read_csv('wine.data')
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

print("\nДанные:")
print("------------------")
print(data.head())
print("\nX:")
print("------------------")
print(X.head())
print("\ny:")
print("------------------")
print(y.head())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

kMeans = list()
for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=X, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

print(kMeans)
m = max(kMeans)
indexM = kMeans.index(max(kMeans))+1
print("\n без масштабирования признаков best k= ", indexM)
print(np.round(m, decimals=2))

scaledX = scale(X)
print("\nX масштабированное :")
print("------------------")
print(scaledX[:5])

kMeans = list()
for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=scaledX, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

print(kMeans)
m = max(kMeans)
indexM = kMeans.index(max(kMeans))+1
print("\n с масштабированием признаков best k= ", indexM)
print(np.round(m, decimals=2))
