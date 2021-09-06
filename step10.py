# 파이썬으로 테스트할때 unittest를 사용하면 편하다.
import unittest
import numpy as np
from step09 import *
class SquareTest(unittest.TestCase):
# test로 시작하는 메서드를 만들고 그 안에 테스트할 내용을 적는다.
# 아래는 square함수의 출력이 기댓값과 같은지 확인한다.
    def test_forward(self):
        x=Variable(np.array(2.0))
        y=square(x)
        expected=np.array(4.0)
        self.assertEqual(y.data,expected)
    def test_backward(self):
        x=Variable(np.array(3.0))
        y=square(x)
        y.backward()
        expected=np.array(6.0)
        self.assertEqual(x.grad,expected)
if __name__=="__main__":
    unittest.main()