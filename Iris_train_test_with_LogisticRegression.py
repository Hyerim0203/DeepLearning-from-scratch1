#!/usr/bin/env python
# coding: utf-8

# In[26]:


# 모듈 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LogisticRegression_class_B635344_이혜림 import Multiple_LogisticRegression, Binary_LogisticRegression 


# In[3]:


# 데이터 로드
from sklearn.datasets import load_iris

iris = load_iris()
iris_x = iris.data # iris data input
iris_y = iris.target # iris target(label)
iris_y_name = iris.target_names # iris target name
iris_feature_names = iris.feature_names


# sklearn을 이용해 train, test data 7:3으로 분류
X_train,X_test,y_train, y_test = train_test_split(iris_x,iris_y, test_size=0.3, random_state=11)


# # binary-class classification

# In[125]:


# 학습
# hyperparameters
lrs = [0.5,0.1,0.01]
epochs = [100,300,500]

# y의 label 마다 각 lr를 갖는 Binary model 생성
model_binary_lrs = {label:[] for label in np.unique(iris_y)}
# y의 label 마다 각 lr를 갖는 cost를 담을 costs생성
costs={label:[] for label in np.unique(iris_y)}
# y의 label 마다 각 lr를 갖는 정확도를 담을 accuracy생성
accuracys = {label:[] for label in np.unique(iris_y)}

# model 생성
for label in model_binary_lrs.keys():
    for lr in lrs: 
        model_binary_lrs[label].append(Binary_LogisticRegression(300,label,learning_rate=lr)) # Epoch : 300

        
# train & test
for label , models in model_binary_lrs.items():
    for idx, m in enumerate(models):
        print("Label={:2}, Learing_rate = {}".format(label, lrs[idx]))
        m._train(X_train,y_train)
        accuracy = m._accuracy(X_test,y_test)
        accuracys[label].append(accuracy)
        costs[label].append(pd.DataFrame(m.total_cost).T)
    print("")


# In[126]:


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
    
        


# In[201]:


# y의 label 마다 각 epoch를 갖는 Binary model 생성
model_binary_epochs = {label:[] for label in np.unique(iris_y)}
# y의 label 마다 각 epoch를 갖는 정확도를 담을 accuracy생성
accuracys = {label:[] for label in np.unique(iris_y)}
# y의 label 마다 각 epoch를 갖는 cost를 담을 costs생성
costs={label:[] for label in np.unique(iris_y)}

# learning_rate = 0.1 model 생성
for label in model_binary_epochs.keys():
    for epoch in epochs: 
        model_binary_epochs[label].append(Binary_LogisticRegression(epoch,label,learning_rate=0.1))

# train & test
for label , models in model_binary_epochs.items():
    for idx, m in enumerate(models):
        print("Label={:2}, Epoch = {}".format(label, epochs[idx]))
        m._train(X_train,y_train)
        accuracy = m._accuracy(X_test,y_test)
        accuracys[label].append(accuracy)
        costs[label].append(pd.DataFrame(m.total_cost).T)
    print("")


# In[204]:


# cost 시각화 및 정확도 출력
for label, a in accuracys.items(): # label
    print(f"Label = {label}")
    for idx, a in enumerate(a): # 해당 label에서의 lr을 달리하는 cost들
        print("Epoch:{:<3}, Accuracy:{:.2%}".format(epochs[idx],accuracys[label][idx]))
        plt.plot(costs[label][idx], label=epochs[idx])
        plt.title(f"Label={label}")
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.legend()
    plt.show()


# # Multiple-class Classification

# In[167]:


# 학습
#hyperparameters

# epoch : 300, learning_rate=0.1 모델 생성
model_multiple=Multiple_LogisticRegression(300,learning_rate=0.1) 
        
# train & test
model_multiple._train(X_train,y_train)
accuracy = model_multiple._accuracy(X_test,y_test)
costs=pd.DataFrame(model_multiple.total_cost).T


# In[168]:


# cost 시각화 및 정확도 출력
plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.show()
print("Accuracy:{:.2%}".format(accuracy))


# In[187]:


# 학습
#hyperparameters

# epoch : 300, learning_rate=0.5 모델 생성
model_multiple=Multiple_LogisticRegression(300,learning_rate=0.5) 
        
# train & test
model_multiple._train(X_train,y_train)
accuracy = model_multiple._accuracy(X_test,y_test)
costs=pd.DataFrame(model_multiple.total_cost).T


# In[188]:


# cost 시각화 및 정확도 출력
plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.show()
print("Accuracy:{:.2%}".format(accuracy))

