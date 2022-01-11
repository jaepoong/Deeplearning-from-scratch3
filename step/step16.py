#  복잡한 계산 그래프 (이론편)
# 변수와 함수에 세대를 설정한다.
import numpy as np

from step09 import as_array

class Variable():
    # 입력 및 초기화
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation=0

    # creator를 추가한다.
    def set_creator(self, func):
        self.creator = func
        self.generation=func.generation+1
    # 역전파의 미분값 재귀호출
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[]
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs:
            f=funcs.pop()
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx

                if x.creator is not None:
                    add_func(x.creator)
    def clear_grad(self):
        self.grad=None
class Function:
    def __call__(self,*inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
        self.generation=max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=outputs
        return outputs if len(outputs)> 1 else outputs[0]
    def forward(self,x):
        raise NotImplementedError()
    def backward(self,gy):
        raise NotImplementedError()
class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    def backward(self,gy):
        return gy,gy
def add(x0,x1):
    return Add()(x0,x1)
class Square(Function):
    def forward(self,x):
        y=x**2
        return y
    def backward(self,gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx
def square(x0):
    return Square()(x0)
if __name__=="__main__":
    x=Variable(np.array(2.0))
    a=square(x)
    y=add(square(a),square(a))
    y.backward()
    print(y.data)
    print(x.grad)