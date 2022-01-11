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
x=Variable(np.array(2.0))
y=f(x)
y.backward(create_graph=True)
print(x.grad)


gx=x.grad
x.clear_grad()
gx.backward()
print(x.grad)