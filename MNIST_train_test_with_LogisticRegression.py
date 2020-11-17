#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 모듈 임포트 및 데이터 로드
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from LogisticRegression_class_B635344_이혜림 import Multiple_LogisticRegression, Binary_LogisticRegression
mnist = fetch_openml('mnist_784')


# In[7]:


mnist_x = mnist.data/255 # X(값 범위가 0~255이기 때문에 255로 나눠줌으로써 0~1로 정규화)
mnist_y = mnist.target # Y
mnist_y=mnist_y.astype("int32") # mnist_y 타입을 object에서 int로 변경

# sklearn의 패키지를 사용해서 train set 과 test set을 분류
X_train, X_test, y_train, y_test = train_test_split(mnist_x,mnist_y,test_size=0.3, random_state=11)


# # Binary-class Classification

# In[10]:


# 학습
# hyperparameters
lrs = [0.5,0.1,0.05]
epochs = [100,300,500]

# y의 label 마다 각 lr를 갖는 Binary model 생성
model_binary_lrs = {label:[] for label in np.unique(mnist_y)}
# y의 label 마다 각 lr를 갖는 cost를 담을 costs생성
costs={label:[] for label in np.unique(mnist_y)}
# y의 label 마다 각 lr를 갖는 정확도를 담을 accuracy생성
accuracys = {label:[] for label in np.unique(mnist_y)}

for label in model_binary_lrs.keys():
    for lr in lrs: 
        model_binary_lrs[label].append(Binary_LogisticRegression(300,label,learning_rate=lr)) # Epoch : 300

        
# train
for label , models in model_binary_lrs.items():
    for idx, m in enumerate(models):
        print("Label={:2}, Learing_rate = {}".format(label, lrs[idx]))
        m._train(X_train,y_train)
        accuracy = m._accuracy(X_test,y_test)
        accuracys[label].append(accuracy)
        costs[label].append(pd.DataFrame(m.total_cost).T)
    print("")


# In[11]:


# cost 시각화 및 정확도 출력
for label, cs in costs.items(): # label 마다의 cost들
    print(f"Label = {label}")
    for idx, c in enumerate(cs): # 해당 label에서의 lr을 달리하는 cost들
        print("Learning rate:{:<4}, Accuracy:{:.2%}".format(lrs[idx],accuracys[label][idx]))
        plt.plot(c, label=lrs[idx])
        plt.title(f"Label={label}")
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.legend()
    plt.show()


# In[13]:


# y의 label 마다 각 epoch를 갖는 Binary model 생성
model_binary_epochs = {label:[] for label in np.unique(mnist_y)}
# y의 label 마다 각 epoch를 갖는 정확도를 담을 accuracy생성
accuracys = {label:[] for label in np.unique(mnist_y)}

# learning rate = 0.5 model 생성
for label in model_binary_epochs.keys():
    for epoch in epochs: 
        model_binary_epochs[label].append(Binary_LogisticRegression(epoch,label,learning_rate=0.5))

# train & test
for label , models in model_binary_epochs.items():
    for idx, m in enumerate(models):
        print("Label={:2}, Epoch = {}".format(label, epochs[idx]))
        m._train(X_train,y_train)
        accuracy = m._accuracy(X_test,y_test)
        accuracys[label].append(accuracy)
    print("")


# In[14]:


# 정확도 출력
for label, a in accuracys.items(): # label
    print(f"Label = {label}")
    for idx, a in enumerate(a): # 해당 label에서의 lr을 달리하는 cost들
        print("Epoch:{:<3}, Accuracy:{:.2%}".format(epochs[idx],accuracys[label][idx]))


# # Multiple-Class Classification

# In[35]:


# 학습
# hyperparameters

# epoch : 500, learning_rate=0.5 모델 생성
model_multiple=Multiple_LogisticRegression(500,learning_rate=0.5) 
        
# train & test
model_multiple._train(X_train,y_train)
accuracy = model_multiple._accuracy(X_test,y_test)
costs=pd.DataFrame(model_multiple.total_cost).T


# In[36]:


# cost 시각화 및 정확도 출력
plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.show()
print("Accuracy:{:.2%}".format(accuracy))


# In[44]:


# 학습
# hyperparameters

# epoch : 300, learning_rate=0.5 모델 생성
model_multiple=Multiple_LogisticRegression(300,learning_rate=0.5) 
        
# train & test
model_multiple._train(X_train,y_train)
accuracy = model_multiple._accuracy(X_test,y_test)
costs=pd.DataFrame(model_multiple.total_cost).T


# In[45]:


# cost 시각화 및 정확도 출력
plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.show()
print("Accuracy:{:.2%}".format(accuracy))

