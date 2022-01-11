if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가
    
import numpy as np
from dezero import Function
from dezero import Variable
import math

def rosenbrock(x0,x1):
    y=100*(x1-x0**2)**2+(1-x0)**2
    return y

x0=Variable(np.array(0.0))
x1=Variable(np.array(2.0))
lr=0.001
iters=10000
for i in range(iters):
    print(x0,x1)
    y=rosenbrock(x0,x1)
    x0.clear_grad()
    x1.clear_grad()
    y.backward()
    x0.data-=lr*x0.grad
    x1.data-=lr*x1.grad