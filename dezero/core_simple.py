# Dezero 코어 모듈
import numpy as np
import weakref
import contextlib
class Variable():
    '''
    data = numpy 데이터
    grad = gradient
    creator = Function
    generator = 몇번째 세대인가
    '''
    __array_priority__=200 # nd.array의 우선순위를 뒤편으로
    # 입력 및 초기화
    def __init__(self, data,name=None):
        # 입력데이너 numpy 아닐 시 TypeError 발생
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))
        self.data = data # 변수값
        self.name=name
        self.grad = None # gradient
        self.creator = None # 해당 변수의 Function
        self.generation=0 # 해당 Variable의 세대 : 역전파 미분시에 데이터 참조 순서를 보인다.

    # creator를 추가한다.
    def set_creator(self, func):
        self.creator = func
        self.generation=func.generation+1
    # 역전파의 미분값 재귀호출
    def backward(self,retain_grad=False):
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
            if not retain_grad:
                for y in f.outputs:
                    y().grad=None
    def clear_grad(self): # grad 초기화
        self.grad=None
    # 넘파이의 여러 메서드 구현.
    def shape(self):
        return self.data.shape
    def ndim(self):
        return self.data.ndim
    def size(self):
        return self.data.size
    def dtype(self):
        return self.data.dtype
    # 객체수를 알려주는 표준함수.  -> 파이썬의 특별한 의미의 메서드 __init__같은건 밑줄 두개로 감싼 이름을 사용.
    def __len__(self):
        return len(self.data)
    # print 표준함수 변경
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'
    # *가능하게 변경
class Function:
    def __call__(self,*inputs):
        inputs=[as_variable(x) for x in inputs]
        xs=[x.data for x in inputs] #input에서 데이터 추출.
        ys=self.forward(*xs) # function 진행
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys] # 출력을 Variable값으로
        if Config.enable_backprop:# 역전파 안할 시에 input 저장 안함.
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
class Config: # Config.enable_backprop=False로 설정하면 역전파 안한다.
    enable_backprop=True # 역전파 활성 비활성을 위한 함수.
class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    def backward(self,gy):
        return gy,gy
class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        return gy*x1,gy*x0
class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self,gy):
        return -gy
class Sub(Function):
    def forward(self,x0,x1):
        y=x0-x1
        return y
    def backward(self,gy):
        return gy,-gy
class Div(Function):
    def forward(self,x0,x1):
        y=x0/x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        return gx0,gx1
class Pow(Function):
    def __init__(self,c):
        self.c=c
    def forward(self,x):
        y=x**self.c
        return y
    def backward(self,gy):
        x=self.inputs[0].data
        c=self.c
        gx=c*x**(c-1)*gy
        return gx
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)
def no_grad():
    return using_config('enable_backprop',False)
def as_array(x):
    if np.isscalar(x):# 입력 데이터가 np.float같은 스칼라 타입인지 확인하는 함수.
        return np.array(x)
    return x
def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)
def add(x0,x1):
    x1=as_array(x1) # 기본 자료형도 대입연산 가능하게 변경
    return Add()(x0,x1)
def mul(x0,x1):
    x1=as_array(x1)
    return Mul()(x0,x1)
def neg(x):
    return Neg()(x)
def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)
def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)
def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)
def pow(x,c):
    return Pow(c)(x)
Variable.__pow__=pow
Variable.__truediv__=div
Variable.__rtruediv__=rdiv
Variable.__sub__=sub
Variable.__mul__=mul
Variable.__add__=add
Variable.__radd__=add
Variable.__rmul__=mul
Variable.__neg__=neg
Variable.__rsub__=rsub # 덧샘과 곱셈은 좌우항의 순서를 바꿔도 변함없다. 그러나 뺄 나눗셈은 둘다 구별해야 하기에 rsub를 따로 구현해야한다.