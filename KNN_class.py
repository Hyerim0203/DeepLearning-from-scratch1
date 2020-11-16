#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np

class KNN(object):
    def __init__(self):
        pass
    def train(self,x,y):
        # train data의 input data와 label을 기억
        self.train_X = x
        self.train_y = y
        
    def test(self,x,method="simple",k=1):
        self.test_X = x
        self.k = k
        self.num_data = self.test_X.shape[0]
        
        # matrix 곱연산을 이용한 유클리드 거리 계산
        squre1 = (self.test_X**2).sum(axis=1) # test data 제곱 부분
        squre2 = (self.train_X**2).sum(axis=1) # train data 제곱 부분
        matmul = self.test_X.dot(self.train_X.T)*-2 # test data와 train 데이터의 -2ab 곱 부분
        squre1=squre1.reshape(self.num_data,1) # 열로의 브로드 캐스팅을 위한 reshape
        dis = squre1+matmul
        dis = squre2+dis
        dis = np.sqrt(dis)
        
        # 거리가 가장 가까운 k개의 data의 투표에 의한 label predict
        if method=="simple":
            predicted = self.majority_vote(dis)
        elif method=="weighted":
            predicted = self.weighted_majority_vote(dis)
            
        return predicted, dis
    
    def majority_vote(self,dis):
        # k개의 투표애 대한 가중치 동일
        predicted = np.zeros(self.num_data,dtype="int32") # 각 test_data에 대한 예측 label
        for idx, data in enumerate(dis):
            hubo=self.train_y[data.argsort()[:self.k]]
            label = np.bincount(hubo).argmax()
            predicted[idx]=label
            
        return predicted
    
    def weighted_majority_vote(self,dis):
        # k개의 투표에 대한 가중치 -> 1/distance의 가중치 부여
        predicted = np.zeros(self.num_data,dtype="int32")
        voted = np.zeros(self.train_y.max()+1)
        
        for idx, data in enumerate(dis):
            voted = np.zeros(self.train_y.max()+1)
            hubo_index = data.argsort()[:self.k]
            weighted = np.exp(-data[hubo_index])
            for idx_vote, w in enumerate(weighted): # 가중치에 의한 투표
                voted[self.train_y[hubo_index[idx_vote]]]+=w
            label = voted.argmax()
            predicted[idx]=label
            
        return predicted
            

