import numpy as np
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
class Variable:
    # 입력 및 초기화
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None
    # creator를 추가한다.
    def set_creator(self,func):
        self.creator=func
    # 역전파의 미분값 재귀호출
    def backward(self):
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()# 함수를 가져온다 y에가까운쪽부터
            x,y=f.input , f.output # 입력과 출력을 가져온다.
            x.grad=f.backward(y.grad) # 출력을 입력으로 역전파 실시.
            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.
if __name__=="__main__":
    A=Square()
    B=Exp()
    C=Square()

    x=Variable(np.array(0.5))
    a=A(x)
    b=B(a)
    y=C(b)

    #역전파
    y.grad=np.array(1.0)
    y.backward()
    print(x.grad)