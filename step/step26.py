if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가

from dezero import Variable
from dezero.utils import _dot_var, plot_dot_graph
from dezero.utils import _dot_func,_get_dot_graph
import numpy as np
def goldstein(x,y):
    z=(1+(x+y+1))**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)
    return z

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=goldstein(x,y)
z.backward()

x.name="x"
y.name="y"
z.name="z"
plot_dot_graph(z,verbose=False, to_file='goldstein.png')