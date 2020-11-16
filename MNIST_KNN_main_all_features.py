#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 데이터 불러오기

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784') # sklearn 이용해서 데이터 불러오기
"""
mnist key 설명
data : 7만개의 28*28인 이미지를 자동으로 784의 1차원으로 저장된 데이터, type : float64, shape : (70000,784)
target : data의 label(0~9), type : object, shape : (70000,)
"""


# In[2]:


mnist_x = mnist.data # X
mnist_y = mnist.target # Y
mnist_y=mnist_y.astype("int32") # 문자열로 저장되어 있는 것을 int 형식으로 변경


# In[3]:


# 전체 데이터에서 test data로 사용할 데이터의 index를 랜덤으로 10000개 추출
np.random.seed(seed=50)
test_idx = np.random.choice(mnist_x.shape[0],10000, replace=False)


# In[4]:


# train data와 test data를 분리
bool_idx = np.zeros(mnist_x.shape[0], dtype=bool) 
bool_idx[test_idx]=True # test index 값에서만 True인 불리언 배열
X_train,X_test,y_train,y_test = mnist_x[~bool_idx], mnist_x[bool_idx], mnist_y[~bool_idx], mnist_y[bool_idx]

"""
X_train shape : (60000,784)
X_test shape  : (10000,784)
y_train shape : (60000,)
y_test shape  : (10000,)
"""


# In[5]:


# 시각화

for i in range(0,10):
    cv_list=mnist_x[mnist_y==i][0:5]
     
    for idx,image in enumerate(cv_list):
        plt_idx = 10*idx+i+1 # 그림이 위치할 위치
        plt.subplot(5,10,plt_idx) 
        plt.imshow(image.astype("uint8").reshape(28,28),cmap="gray")
        plt.axis("off")
        if idx==0:
            plt.title(f"{i}")
plt.show()


# In[6]:


# normalize
# 0~1 사이에 값이 위치하기 위해 X data(0~255)의 Maximum 값인 255를 나누어 줌.
X_train=X_train/255
X_test=X_test/255


# In[7]:


# 데이터 분류
from knn_class_b635344_이혜림 import KNN
model = KNN()

# Train & Test with all features
model.train(X_train,y_train)


# In[8]:


# 10000개의 test data 중에 random으로 1000개 선택
np.random.seed(seed=100)
idx=np.random.choice(len(X_test),1000,replace=False)

cv_predicted={3:[],5:[],7:[],9:[],10:[]}
cv_accuracy={3:[],5:[],7:[],9:[],10:[]}
for k in cv_accuracy.keys():
    predict1,_ = model.test(X_test[idx],k=k)                   # majority vote predicted
    predict2,_ = model.test(X_test[idx],k=k,method="weighted") # weighted vote predicted
    
    cv_predicted[k].append(predict1)
    cv_predicted[k].append(predict2)
    
    cv_accuracy[k].append(sum(predict1==y_test[idx])/1000) # majority vote accuracy
    cv_accuracy[k].append(sum(predict2==y_test[idx])/1000) # weighted vote accuracy


# In[9]:


# 예측결과 출력
label_name=["0","1","2","3","4","5","6","7","8","9"]
for k,m in cv_predicted.items():
    print("K = {}".format(k))
    for i in range(1000):
        p1 =label_name[m[0][i]]
        p2 = label_name[m[1][i]]
        y=y_test[idx]
        label=label_name[y[i]]
        print("{:4}th  simple voted : {}, weighted voted : {}, True label : {}".format(idx[i],p1,p2,label))
        


# In[10]:


# 정확도 출력
for k,a in cv_accuracy.items():
    print("K = {}\nsimple vote accuracy : {:.2%} weighted vote accuracy : {:.2%}".format(k,a[0],a[1]))

