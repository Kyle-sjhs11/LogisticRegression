# Logistic Regression
'''
胡开智 222021335210071
date: 11/30/2022
IDE:Visual Studio Code
Python:3.9
'''
# 加载相关的包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import metrics

# 加载数据集
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 归一化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练模型
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# 预测数据
y_pred = classifier.predict(X_test)

# 将编码转换为类别
cols = y_test.shape[0]
y = []
pre = []
for i in range(cols):
    if y_test[i] == 1:
        y.append('Yes')
    elif y_test[i] == 0:
        y.append('No')
for j in range(cols):
    if y_pred[j] == 1:
        pre.append('Yes')
    elif y_pred[j] == 0:
        pre.append('No')  
    else:
        pre.append('unknown')

# 混淆矩阵
sns.set()
cm = confusion_matrix(y, pre, labels=["Yes","No"])
df = pd.DataFrame(cm, index=["Yes","No"], columns=["Yes","No"])
ax = sns.heatmap(df, annot=True)
ax.set_xlabel("predict")
ax.set_ylabel("true")
plt.show()
# 分类报告
print(metrics.classification_report(y, pre, zero_division=True))

# 训练结果可视化
# 在训练集上的表现
X_set, y_set = X_train, y_train #训练集表现
#X_set, y_set = X_test, y_test  #测试集表现
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
print(X1,X2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 在测试集上的表现
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



'''
# 不调用包
class Logistic_Regression:
    def __init__(self,traindata,alpha=0.001,circle=1000,batchlength=40):
        self.traindata=traindata #训练数据集
        self.alpha=alpha #学习率
        self.circle=circle #学习次数
        self.batchlength=batchlength #把3349个数据分成多个部分,每个部分有batchlength个数据
        self.w=np.random.normal(size=(3,1)) #随机初始化参数w
    def data_process(self):
        #做随机梯度下降,打乱数据顺序,并把所有数据分成若干个batch
        np.random.shuffle(self.traindata)
        data=[self.traindata[i:i+self.batchlength]
              for i in range(0,len(self.traindata),self.batchlength)]
        return data
    def train(self):
        #根据损失函数(1)来进行梯度下降，这里采用随机梯度下降
        for i in range(self.circle):
            batches=self.data_process()
            print('the {} epoch'.format(i)) #程序运行时显示执行次数
            for batch in batches:
                d_w=np.zeros(shape=(3,1)) #用来累计w导数值
                for j in batch: #取batch中每一组数据
                    x0=np.r_[j[0:2],1] #把数据中指标取出,后面补1
                    x=np.mat(x0).T #转化成列向量
                    y=j[2] #标签
                    dw=(self.sigmoid(self.w.T*x)-y)[0,0]*x
                    d_w+=dw
                self.w-=self.alpha*d_w/self.batchlength
                #动态可视化
            w=regr.w
            w1=w[0,0]
            w2=w[1,0]
            w3=w[2,0]
            x=np.arange(190,500)
            y=-w1*x/w2-w3/w2
            plt.plot(x,y)
            plt.show()
   
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
 
    def predict(self,x):
        #测试新数据属于哪一类,x是2维列向量
        s=self.sigmoid(self.w.T*x)
        if s>=0.5:
            return 1
        elif s<0.5:
            return 0
        
# 加载数据集并分割打包
dataset = pd.read_csv('Social_Network_Ads.csv')
x1 = dataset.iloc[:, [2]].values
x2 = dataset.iloc[:, [3]].values
label = dataset.iloc[:, 4].values
train_data=list(zip(x1,x2,label))

regr=Logistic_Regression(traindata=train_data)
regr.train()
'''