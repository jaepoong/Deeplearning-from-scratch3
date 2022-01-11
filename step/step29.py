if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가
    
import numpy as np
from dezero import Function
from dezero import Variable
import math

def f(x):
    y=x**4-2*x**2
    return y

def gx2(x):
    return 12*x**2-4

x=Variable(np.array(2.0))
iters=10

for i in range(iters):
    print(i,x)
    y=f(x)
    x.clear_grad()
    y.backward()
    x.data-=x.grad/gx2(x.data)