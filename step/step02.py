import numpy as np
# 변수를 낳는 함수
from step01 import Variable
class Function:
    def __call__(self,input):# 파이썬의 특수 메서드이다. f=Func()형태로 저장후 f(...)로 __call__메서드 호출한다.
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        return output
    def forward(self,x):# 수식에 관한 부분은 forward에서
        raise NotImplementedError()# 상속을 해야 구현할 수 있게 에러 발생시킨다.
class Square(Function):
    def forward(self,x):
        return x**2

if __name__=="__main__":
    x=Variable(np.array(10))
    f=Square()
    y=f(x)
    print(type(y))
    print(y.data)