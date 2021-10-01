# 패키지로 정의
# dezero 임포트하기
if '__file__' in globals(): #터미널에서 명령 실행하면 __file__변수가 정의되어 있다.
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),"..")) #부모디렉터리를 모듈 검색 경로에 추가
import numpy as np
from dezero import Variable

x=Variable(np.array(1.0))
y=(x+3)**2
y.backward()
print(y,x.grad)