import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
import tkinter
#%%
np.random.seed(0)
#%%
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
HeartDisease = pandas.read_csv('processed.cleveland.csv', header=None, names=names)
#%%
median=HeartDisease['ca'].median()
HeartDisease['ca'].fillna(median,inplace=True)
median=HeartDisease['thal'].median()
HeartDisease['thal'].fillna(median,inplace=True)
#HeartDisease.info()
#%%
label = HeartDisease['num']
data = HeartDisease.drop(['num'],axis=1)
#%%
if data.shape[0] == label.shape[0]:
    print('The two result files have written to the Output folder! ')
else:
    print('Sample number not equal to label number')
    exit(-1)
#%%
x_train , x_test , y_train , y_test =train_test_split(data, label, test_size=0.25, train_size=0.75, random_state=42)
#%%
clf = RandomForestClassifier(n_estimators=100,max_depth=10)
clf = clf.fit(x_train,y_train)
result = clf.score(x_test, y_test)
print(result)
#%%
clf = BaggingClassifier(base_estimator=RandomForestClassifier(max_depth=10))
clf = clf.fit(x_train,y_train)
result = clf.score(x_test, y_test)
print(result)
#%%
answer = tkinter.Tk()
answer.title('预测准确率')
answer.mainloop()