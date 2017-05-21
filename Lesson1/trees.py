# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:43:27 2017

Учимся обучать решающие деревья

@author: Musimusimus
"""

import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


print ("------------------------ dz1---------------------")


# пример из учебника:
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)

# последовательно обрезаем данные, чтобы получить нужную выборку
data = data[['Survived','Pclass','Fare','Age','Sex']]
# выкидываем строки с пустыми ячейками:
data = data.dropna(axis=0, how='any')
# определяем целевые листья-значения
target = data[['Survived']]
# признаки:
data = data[['Pclass','Fare','Age','Sex']]

print(data)

# кодируем Sex в числа 1,0:
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex)

print(data)

# обучаем дерево со случайным коэфициентом из методички:
tree = DecisionTreeClassifier(random_state=241)
tree.fit(data, target)

# находим важность признаков для предсказания целевой переменной:
importances = tree.feature_importances_

print (importances)



