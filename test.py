from time import time
class test(object):
    times=[]
    
    def __init__(self,x):
        self.x = x

    def fun1(self):
        s = time()
        result = self.x**5
        e = time()
        test.times.append(e-s)
        return result

ins1 = test(2)
ins1.fun1()
ins2 = test(2)
ins2.fun1()
print(test.times)