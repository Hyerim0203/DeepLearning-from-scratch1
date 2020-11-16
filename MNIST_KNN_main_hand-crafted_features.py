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


# In[8]:


# train&test with hand crafted features
# 각 데이터를 4개의 features씩 나누어서 4개의 평균값을 대표값으로 하여 차원을 축소.

# train data 차원 축소
prj_X_train=np.zeros((len(X_train),196)) # 차원축소된 데이터를 담을 (60000,196) 형태의 zero array 생성
for i in range(len(X_train)):
    prj_X_train[i]=X_train[i].reshape(-1,4).mean(axis=1) # 4개의 feature 씩 묶어서 해당 그룹의 평균을 zero array에 추가

# test data 차원 축소
prj_X_test=np.zeros((len(X_test),196)) # 차원축소된 데이터를 담을 (10000,196) 형태의 zero array 생성
for i in range(len(X_test)):
    prj_X_test[i]=X_test[i].reshape(-1,4).mean(axis=1) # 4개의 feature 씩 묶어서 해당 그룹의 평균을 zero array에 추가


# In[9]:


# train
model.train(prj_X_train,y_train)

# 해당 차원축소에 대해 잘 작동하는지 10개의 set으로 확인
np.random.seed(seed=200)
pre_idx = np.random.choice(len(X_test),10,replace=False)

pre_predict,_= model.test(prj_X_test[pre_idx],k=10,method="weighted")
print(sum(pre_predict==y_test[pre_idx])/10) # 정확도 출력


# In[10]:


# test
# 차원축소를 하지 않았을 때 사용한 동일한 1000개의 random한 index를 추출하여 test data추출에 사용
np.random.seed(seed=100)
idx=np.random.choice(len(X_test),1000,replace=False)

prj_cv_predicted={3:[],5:[],7:[],9:[],10:[]}
prj_cv_accuracy={3:[],5:[],7:[],9:[],10:[]}
for k in prj_cv_accuracy.keys():
    predict1,_ = model.test(prj_X_test[idx],k=k)                   # majority vote predicted
    predict2,_ = model.test(prj_X_test[idx],k=k,method="weighted") # weighted vote predicted
    
    prj_cv_predicted[k].append(predict1)
    prj_cv_predicted[k].append(predict2)
    
    prj_cv_accuracy[k].append(sum(predict1==y_test[idx])/1000) # majority vote accuracy
    prj_cv_accuracy[k].append(sum(predict2==y_test[idx])/1000) # weighted vote accuracy


# In[11]:


# 예측결과 출력
label_name=["0","1","2","3","4","5","6","7","8","9"]
for k,m in prj_cv_predicted.items():
    print("K = {}".format(k))
    for i in range(1000):
        p1 =label_name[m[0][i]]
        p2 = label_name[m[1][i]]
        y=y_test[idx] # x_test 배열과 같은 순서 배열로 맞춰줌
        label=label_name[y[i]]
        print("{:4}th  simple voted : {}, weighted voted : {}, True label : {}".format(idx[i],p1,p2,label))
        


# In[12]:


# 정확도 출력
for k,a in prj_cv_accuracy.items():
    print("K = {}\nsimple vote accuracy : {:.2%} weighted vote accuracy : {:.2%}".format(k,a[0],a[1]))

