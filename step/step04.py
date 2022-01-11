#수치 미분
#중앙 차분으로 하여 h=0.0001로 진행한다. : 근사미분
from step01 import Variable
from step02 import Square
import numpy as np
from step03 import Exp

def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(eps*2)
def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))
if __name__=="__main__":
    a=Square()
    x=Variable(np.array(2.0))
    dy=numerical_diff(a,x)
    print(dy)

    x=Variable(np.array(0.5))
    dy=numerical_diff(f,x)
    print(dy)
# 수치 미분의 문제점은 매우 계산량이 많고 오차가 있다.