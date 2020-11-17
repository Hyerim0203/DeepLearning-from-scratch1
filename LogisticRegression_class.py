#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[1]:


class Multiple_LogisticRegression:
    def __init__(self, epoch, learning_rate=0.01):
        # hyperparameters
        self.epoch = epoch
        self.lr = learning_rate
        
        # epcch 마다의 cost 저장
        self.total_cost = {}
        
    def _sigmoid(self,z):
        # 분모 exp 에서의 overflow를 방지
        eMin = -np.log(np.finfo(type(0.1)).max)
        zSafe = np.maximum(z,eMin)
        return 1/(1+np.exp(-zSafe))
        
    def _train(self, X, y):
        # parameters
        data_num = X.shape[0] # 데이터 개수
        feature_num = X.shape[1]+1 # bias term까지 고려한 feature 개수
        label_num = y.max()+1 # label 개수
        
        X_train = np.concatenate((np.ones((data_num,1)),X),axis=1) # X에 bias term을 넣어줌
        Y_train = np.eye(label_num)[y] # one hot encoding
        W = np.random.randn(feature_num,label_num) # 정규분포로 초기화된 W
        
        for epoch in range(self.epoch): # 전체 데이터를 epoch 만큼 반복 학습
            # X,W의 벡터 연산 후, sigmoid 함수에 넣어 score 계산
            score = self._sigmoid(X_train.dot(W))
            
            # cost 계산
            c=Y_train*np.log(score)+(1-Y_train)*np.log(1-score)
            self.cost = (-1/data_num)*np.sum(c, axis=0) # 한 epoch 마다의 cost
            
            # 저장 및 출력
            self.total_cost[epoch]=self.cost
            print("Epoch :{:<4}\nCost : {}".format(epoch, self.cost))
            
            gradient = np.zeros(W.shape) # 모든 W에 대한 gradient            
            
            # 각 label 마다 gradient 계산하여 전체 gradient 계산
            for label in range(label_num):
                # size=(data개수,1) 와 size=(data개수, feature개수) 브로드캐스트 연산
                g = np.zeros((data_num,feature_num))
                g += (score[:,label]-Y_train[:,label]).reshape(-1,1)
                """
                각 data의 feature에 해당하는 x 값과 해당 data의 (y_hat-y) 의 값을 곱해준 후, 
                모든 데이터의 값을 같은 feature 별로 더해줌
                """
                g = np.sum(X_train*g,axis=0) # 해당 label 에 대한 feature의 gradient
                
                gradient[:,label]=g # 전체 gradient에 더해줌
            
            # W update
            W-=1/data_num*(self.lr*gradient) # learging rate에 1/date_num 을 녹이지 않음
            self.W=W
        
    def _predict(self,X_test):
        data_num=X_test.shape[0]
        X_test = np.concatenate((np.ones((data_num,1)),X_test),axis=1) # X에 bias term을 넣어줌
        
        # predict
        score = self._sigmoid(X_test.dot(self.W))
        predicted = score.argmax(axis=1)
        
        return predicted
    
    def _accuracy(self, X_test, y_test):
        predicted = self._predict(X_test) # predict 결과 얻어옴
        accuracy = np.sum(predicted == y_test)/y_test.shape[0]
        
        return accuracy


# In[4]:


class Binary_LogisticRegression:
    def __init__(self, epoch, label, learning_rate=0.01):
        # hyperparameters
        self.epoch = epoch
        self.lr = learning_rate
        self.label = label # label에 대한 binary classification
        
        # epcch 마다의 cost 저장
        self.total_cost = {}
        
    def _sigmoid(self,z):
        # 분모 exp 에서의 overflow를 방지
        eMin = -np.log(np.finfo(type(0.1)).max)
        zSafe = np.maximum(z,eMin)
        return 1/(1+np.exp(-zSafe))
        
    def _train(self, X, y):
        # parameters
        data_num = X.shape[0] # 데이터 개수
        feature_num = X.shape[1]+1 # bias term까지 고려한 feature 개수

        X_train = np.concatenate((np.ones((data_num,1)),X),axis=1) # X에 bias term을 넣어줌
        Y_train = np.array(y==self.label, dtype=np.int32).reshape(-1,1)
        W = np.random.randn(feature_num,1) # 정규분포로 초기화된 W
        
        for epoch in range(self.epoch): # 전체 데이터를 epoch 만큼 반복 학습
            # X,W의 벡터 연산 후, sigmoid 함수에 넣어 score 계산
            score = self._sigmoid(X_train.dot(W))
            
            # cost 계산
            c=Y_train*np.log(score)+(1-Y_train)*np.log(1-score)
            self.cost = (-1/data_num)*np.sum(c) # 한 epoch 마다의 cost
            
            # 저장 및 출력
            self.total_cost[epoch]=[self.cost]
            print("Epoch :{:<4}\nCost : {}".format(epoch, self.cost))
            
            # size=(data개수,1) 와 size=(data개수, feature개수) 브로드캐스트 연산
            g = np.zeros((data_num,feature_num))
            g += (score-Y_train).reshape(-1,1)
            """
            각 data의 feature에 해당하는 x 값과 해당 data의 (y_hat-y) 의 값을 곱해준 후, 
            모든 데이터의 값을 같은 feature 별로 더해줌
            """
            gradient = np.sum(X_train*g,axis=0).reshape(-1,1) # feature의 gradient 

            # W update
            W-=1/data_num*(self.lr*gradient) # learging rate에 1/date_num 을 녹이지 않음
            self.W=W
        
    def _predict(self,X_test):
        data_num=X_test.shape[0]
        X_test = np.concatenate((np.ones((data_num,1)),X_test),axis=1) # X에 bias term을 넣어줌
        
        # predict
        score = self._sigmoid(X_test.dot(self.W))
        predicted = np.array(score>=0.5, dtype=np.int32)
        
        return predicted.reshape(-1,)
    
    def _accuracy(self, X_test, y_test):
        predicted = self._predict(X_test) # predict 결과 얻어옴
        # 각 데이터가 해당 label에 속하는지에 대한 y값 새로 도출
        label_y = np.array(y_test==self.label, dtype=np.int32) 
        accuracy = np.sum(predicted == label_y)/y_test.shape[0]
        
        return accuracy


# In[ ]:




