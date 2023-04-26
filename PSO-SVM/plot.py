import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
heart_data = pd.read_csv('heart.csv')

# 计算相关系数矩阵
corr_matrix = heart_data.corr()

# 绘制热力图
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr_matrix,annot=True)



# 显示图形
plt.show()
