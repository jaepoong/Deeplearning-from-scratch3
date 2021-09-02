# 역전파 자동화
# 함수는 변수를 입출력에 사용하고 변수는 함수를 창조자로 여긴다.
import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None
    def set_creator(self,creator):
        self.creator=creator
    def backward(self):
        f=self.creator
        if f is not None:
            x=f.input
            x.grad=f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self,input):
        x=input
        y=self.forward(x)
        output=Variable(y)
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
if __name__=="__main__":
    A=Square()
    B=Exp()
    C=Square()
    x=Variable(np.array(0.5))
    a=A(x)
    b=B(a)
    y=C(b)

    y.grad=np.array(1.0)
    y.backward()
    print(x.grad)