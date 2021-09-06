import numpy as np
class Function:
    def __call__(self,input):
        x=input
        y=self.forward(x)
        output=Variable(as_array(y))
        output.set_creator(self)
        self.input=input
        self.output=output
        return output

    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()
class Square(Function):
    def forward(self, x):
        y = x.data**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
class Exp(Function):
    def forward(self, x):
        y = np.exp(x.data)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
def square(x):
    f=Square()
    return f(x)
def exp(x):
    f=Exp()
    return f(x)
def as_array(x):
    if np.isscalar(x):# 입력 데이터가 np.float같은 스칼라 타입인지 확인하는 함수.
        return np.array(x)
    return x
# Backward 메서드 간소화
class Variable:
    # 입력 및 초기화
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))
        self.data=data
        self.grad=None
        self.creator=None
    # creator를 추가한다.
    def set_creator(self,func):
        self.creator=func
    # 역전파의 미분값 재귀호출
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 함수를 가져온다 y에가까운쪽부터
            x, y = f.input, f.output  # 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)  # 출력을 입력으로 역전파 실시.
            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.

if __name__=="__main__":
    x=Variable(np.array(0.5))
    y=square(exp(square(x)))
    y.backward()
    print(x.grad)