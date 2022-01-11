if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가
    
import numpy as np
from dezero import Function
from dezero import Variable
import math

class Sin(Function):
    def forward(self,x):
        y=np.sin(x)
        return y
    def backward(self,gy):
        x=self.inputs[0].data
        gx=gy*np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x,threshhold=0.0001):
    y=0
    for i in range(100000):
        c=(-1)**i/math.factorial(2*i+1)
        t=c*x**(2*i+1)
        y=y+t
        if abs(t.data)< threshhold:
            break
    return y

x=Variable(np.array(np.pi/4))
y=my_sin(x)
y.backward()
print(y.data)
print(x.grad)