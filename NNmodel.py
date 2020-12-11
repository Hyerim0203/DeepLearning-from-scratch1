# coding: utf-8
# 2020/인공지능/final/B635344/이혜림
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    epsilon=1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    #훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask]=0
        
        return out

    def backward(self, dout):
        dout[self.mask]=0
        dx=dout.copy()

        return dx


class CustomActivation: # Leakly Relu
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """ Max(0.1x, x) 활성화 함수 사용을 통하여 dead Relu문제 해결"""
        self.mask = (x<=(0.1*x))
        out = x.copy()
        out[self.mask] = 0
        
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = (dx[self.mask])*0.1
        
        return dx


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W,b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size

        return dx


class SGD:
    def __init__(self, lr=0.0001):
        self.lr = lr
       
    def update(self, params, grads):
        for p in params:
            # parameter update
            params[p] -= self.lr*grads[p]


            
class CustomOptimizer: # momentum 사용
    def __init__(self, lr=0.0001, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {} # 누적 gradient
        
    def update(self, params, grads):
        if not self.v: 
            for key, val in params.items():
                self.v[key]=np.zeros_like(val)
            
        for key in params.keys():
            # gradient를 누적시켜서 저장
            self.v[key] = self.momentum * self.v[key] - self.lr*grads[key]
            # parameter update
            params[key] += self.v[key]
        
class CustomOptimizer2: # RMSProp
    def __init__(self, lr=0.0001, rho=0.99):
        self.lr = lr
        self.rho = rho
        self.h = None # 이전 gradient 저장
    
    def update(self,params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # Adagrad(gradient의 제곱 합)에서 지수이동평균사용함으로써 오래된 gradient의 영향을 줄여나감
            self.h[key] = (self.h[key]*self.rho)+((1-self.rho)*grads[key]*grads[key])
            # parameter update
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)
        

class Model:
    """
    네트워크 모델 

    """
    def __init__(self, lr=0.0001, layer_num=5, node_num = [16,32,64,128,6], optimizer=CustomOptimizer2, activation = Relu):
        """
        클래스 초기화
        """
        # layer 수 설정(loss layer 제외)
        self.layer_num = layer_num
        # 각 layer 내의 node 수 설정 : 마지막 6으로 고정
        self.node_num = node_num
        # 사용할 activation 함수(default : Relu)
        self.activation = activation
        # 사용할 optimizer(default : RMSprop)
        self.optimizer = optimizer(lr)
        
        self.params = {}
        self.__init_weight()
        self.__init_layer()
       

    def __init_layer(self):
        """
        레이어를 생성
        """
        # layer dictionary
        self.layers = {}
        
        """설정한 layer의 수에 맞게 layer 생성
        layer 기본 구조 : Affine => Activation fuction(default = Relu)
        마지막 layer 기본 구조 : Affine"""
        for l_n in range(self.layer_num):
            num = l_n+1 # layer 번호
            if num != self.layer_num: # 기본 구조 : (Affine -> Activation function)
                self.layers["layer{}".format(num)]=Affine(self.params[f"W{num}"], self.params[f"b{num}"])
                self.layers["activation{}".format(num)]=self.activation()
            elif num == self.layer_num: # (Affine)
                self.layers["layer{}".format(num)]=Affine(self.params[f"W{num}"], self.params[f"b{num}"])
                
        # Loss 계산 layer
        self.last_layer = SoftmaxWithLoss()
                

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 
        weight term -> he 초기화 사용(input node와 output node의 차이가 있기 때문에 input node만 사용하는 것 대신 input node+output node 사용)
        bias term -> zero 초기화
        """
        self.params={}
        # before_node : 각 weight의 행의 개수가 될 변수, 처음에 데이터의 feature 개수로 지정
        before_node = 6
        
        for i in range(self.layer_num):
            # weight -> he 초기화
            # input node와 output node의 차이가 있기 때문에 input node만 사용하는 것 대신 input node+output node 사용
            self.params["W{}".format(i+1)]=np.random.randn(before_node,self.node_num[i]) / np.sqrt((before_node+self.node_num[i])/2)
            
            # bias -> zero로 초기화
            self.params["b{}".format(i+1)]=np.zeros(self.node_num[i]) 
            
            # 전 layer의 output node -> 그 다음 layer의 input node
            before_node = self.node_num[i]

    def update(self, x, t):

        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)
        self.__init_layer()

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """                
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)

        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        self.loss(x,t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout=dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # 결과 저장
        grads = {}
        
        # weight, bias term의 gradient : Affine layer의 각 weight, bias gradient
        for n in range(self.layer_num):
            num = n+1 # layer 번호
            grads[f"W{num}"], grads[f"b{num}"] = self.layers["layer{}".format(num)].dW, self.layers["layer{}".format(num)].db
        self.grads = grads
        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        self.__init_layer()
        pass
