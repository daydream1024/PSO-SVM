import pandas
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import tkinter
from sko.PSO import PSO
import matplotlib.pyplot as plt
#%%
np.random.seed(1)#控制变量
#%%
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
HeartDisease = pandas.read_csv('./processed.cleveland.csv',header=None,names=names)
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
    print('Sample number is equal to label number')
else:
    print('Sample number not equal to label number')
    exit(-1)
#%%
clf=SVC(kernel='rbf')#测试不同svm在不同核函数下的性能
acc  = cross_val_score(clf, data,label, cv=3)#使用三倍交叉验证求准确率，后续同
print(np.mean(acc))
#%%
clf = SVC(kernel='poly')#测试不同svm在不同核函数下的性能
acc = cross_val_score(clf, data, label, cv=3)
print(np.mean(acc))
#%%
clf = SVC(kernel='sigmoid')#测试不同svm在不同核函数下的性能
acc = cross_val_score(clf, data, label, cv=3)
print(np.mean(acc))
#%%
clf = SVC(kernel='linear')#测试不同svm在不同核函数下的性能
acc = cross_val_score(clf, data, label, cv=3)
print(np.mean(acc))
#%%
def plots(position):#定义函数用于生成粒子的位置图
    x = []
    y = []
    for i in range(0,len(position)):
        x.append(position[i][0])
        y.append(position[i][1])
    plt.scatter(x, y, alpha = 0.5,c=var.score,cmap='viridis')  #生成散点图，颜色随准对于位置确度变化，透明度设置为0.5
    plt.colorbar()
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.axis([-0.1,1,-1,11])#定义横坐标和纵坐标的范围
    return plt.show()
#%%
class var:
    count=0
    bestsocre=0.0
    bestpos=[]
    tu=[]
    score=[]
#%%
def func(x):#定义函数，用于计算损失率
    x1,x2=x
    clf = SVC
    socre = cross_val_score(clf(kernel='linear',C=x1, gamma=x2),data, label, cv=3).mean()
    if(socre>var.bestsocre):
        var.bestsocre=socre
        var.bestpos=x
    if var.count%psovar.pop==0:
        print('iter', int(var.count / psovar.pop), 'of', psovar.max_iter)#输出当前轮次
    print(x,socre)
    var.tu.append(x)
    var.score.append(socre)
    if var.count%psovar.pop==psovar.pop-1:#输出每一轮粒子的信息
        print('best position is',var.bestpos,'best score is',var.bestsocre)
        var.bestsocre=0
        plots(var.tu)
        var.score=[]
        var.tu=[]
    var.count+=1
    return 1-socre#返回损失率
#%%
class psovar:#pso算法的各种参数
    max_iter=50#迭代次数
    pop=10#粒子数
    n_dim=2#需要求解的维数
    w=0.7#惯性系数，这个数越大，代表着它不容易更改之前的运动路线，更倾向于探索未知领域。
    c1=0.2#个体加速因子
    c2=0.5#社会加速因子
    lb=[0.001, 0.01]#求解未知数下界
    ub=[1, 10]#求解未知数上界
#%%
pso = PSO(func=func, n_dim=psovar.n_dim, pop=psovar.pop, max_iter=psovar.max_iter, lb=psovar.lb, ub=psovar.ub, w=psovar.w, c1=psovar.c1, c2=psovar.c2)
pso.record_mode = True #记录粒子的历史位置
pso.run()#运行算法
print('best_C and best_gamma is ', pso.gbest_x, 'best_cost is', pso.gbest_y)
#%%
plt.plot(pso.gbest_y_hist)#绘制每一论损失率的折线图
plt.show()
#%%
clf=SVC(kernel='linear',C=pso.gbest_x[0],gamma=pso.gbest_x[1])#在pso算法找到的最优位置上测试svm的准确率
psoacc  = cross_val_score(clf, data,label, cv=3).mean()
print(psoacc)
#%%
if data.shape[0] == label.shape[0]:
    root = tkinter.Tk()
    text = tkinter.Text(root,width=20,height=1)
    text.insert('1.0',psoacc)
    text.pack()
    root.title('ops-svm预测准确率')
    root.geometry('240x120')
    button = tkinter.Button(root, text="确定", command=root.destroy)
    button.place(x=100,y=80)
    root.mainloop()
else:
    root = tkinter.Tk()
    text = tkinter.Text(root,width=20,height=1)
    text.insert('1.0','发生错误')
    text.pack()
    root.title('预测准确率')
    root.geometry('240x120')
    button = tkinter.Button(root, text="确定", command=root.destroy)
    button.place(x=100,y=80)
    root.mainloop()