

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:18:44 2022

@author: hanyuanyuan
"""
from sklearn import tree
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
df = pd.read_csv(r'hepatitis.csv', header=None, index_col=False)#index_col=0
#data = df.values[:-1]
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

d = df
print(d)
da = d.dropna()


print(da )
data = da.T[:-1]
data = da.T[:-1].T
print(data.shape)
print(data)
lable =d.dropna(axis=0).T.values[-1]
print(lable)
print(len(lable))

if data.shape[0] == lable.shape[0]:
    print('The two result files have written to the Output folder! ')
else:
    print('Sample number not equal to label number')
    exit(-1)
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, lable)
model = SelectFromModel(lsvc, prefit=True)
F = model.transform(data)     
print(data)


 
clf = tree.DecisionTreeClassifier() #实例化
clf = clf.fit(F,lable) #训练
result = clf.score(F,lable)
print(F.shape)

clf = tree.DecisionTreeClassifier() #实例化
clf = clf.fit(data,lable) #训练
result = clf.score(data,lable)
print(result)


clf = tree.DecisionTreeClassifier()

x_train,x_test,y_train,y_test= train_test_split(F,lable,test_size=0.3)
 
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
acc  =   clf.score(x_test, y_test)
print(acc)


