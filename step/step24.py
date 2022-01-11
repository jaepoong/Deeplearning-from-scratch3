# 복잡한 함수의 미분
if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가
import numpy as np
from dezero import Variable
def sphere(x,y):
    z=x**2+y**2
    return z
def matyas(x,y):
    z=0.26*(x**2+y**2)-0.48*x*y
    return z
def goldstein(x,y):
    z=(1+(x+y+1))**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)
    return z
x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=sphere(x,y)
z.backward()
print(x.grad,y.grad,z)
x.clear_grad()
y.clear_grad()
z=matyas(x,y)
z.backward()
print(x.grad,y.grad,z)
x.clear_grad()
y.clear_grad()
z=goldstein(x,y)
z.backward() 
print(x.grad,y.grad,z)