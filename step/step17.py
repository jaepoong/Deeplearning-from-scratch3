# 메모리 관리와 순환 참조
# 약한참조를 위한 weakref 모듈. 객체를 참조하되 참조 카운트는 증가시키지 않는 기능.
import weakref
import numpy as np
a=np.array([1,2,3])
b=weakref.ref(a)
print(b(),b) # 약한 참조 접근을 위해선 b()사용
a=None
print(b) # 약한참조이기에 영향을 주지 않는다.
from step09 import as_array

class Variable():
    '''
    data = numpy 데이터
    grad = gradient
    creator = Function
    generator = 몇번째 세대인가
    '''
    # 입력 및 초기화
    def __init__(self, data):
        # 입력데이너 numpy 아닐 시 TypeError 발생
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))
        self.data = data # 변수값
        self.grad = None # gradient
        self.creator = None # 해당 변수의 Function
        self.generation=0 # 해당 Variable의 세대 : 역전파 미분시에 데이터 참조 순서를 보인다.

    # creator를 추가한다.
    def set_creator(self, func):
        self.creator = func
        self.generation=func.generation+1
    # 역전파의 미분값 재귀호출
    def backward(self):
        if self.grad is None:  # 최조 grad값을 1로 설정한다.
            self.grad=np.ones_like(self.data)
        funcs=[] # 함수목록
        seen_set=set()
        # Function 입력하고 세대별 정렬.
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs: # 입력 Function당 세대순
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
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
    def clear_grad(self): # grad 초기화
        self.grad=None

class Function:
    def __call__(self,*inputs):
        xs=[x.data for x in inputs] #input에서 데이터 추출.
        ys=self.forward(*xs) # function 진행
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys] # 출력을 Variable값으로
        self.generation=max([x.generation for x in inputs]) # 세댓값
        for output in outputs: #
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=[weakref.ref(output) for output in outputs] # 약한참조로 변경
        return outputs if len(outputs)> 1 else outputs[0]
    def forward(self,x):
        raise NotImplementedError()
    def backward(self,gy):
        raise NotImplementedError()
class Square(Function): # 함수 설정
    def forward(self,x):
        y=x**2
        return y
    def backward(self,gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx
def square(x0): # 파이썬사용 방식으로 변경.
    return Square()(x0)
if __name__=="__main__":
    x=Variable(np.random.randn(10000))
    y=square(square(x))
    print(x.data[0],y.data[0])