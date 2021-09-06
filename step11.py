# 자연스러운 코드로
import numpy as np
# 가변길이 함수
from step09 import Variable, as_array
import numpy as np

class Function:
    def __call__(self,*inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
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
def add(x0,x1):
    return Add()(x0,x1)

if __name__=="__main__":
    x0=Variable(np.array(2))
    x1=Variable(np.array(3))
    f=Add()
    y=add(x0,x1)
    print(y.data)