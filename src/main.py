import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2
def f_diff(x):
    return 2*x
class enviroment():
    def __init__(self,var=[0.1,0.1],h=0.000001,mu0=10,f=f,f_difftmp=f_diff) -> None:
        self.var=var# [x_var,y_var]
        self.h=h#　更新幅
        self.mu0=mu0#事前分布の平均
        self.mu=mu0+np.random.randn()*np.sqrt(self.var[0])# 隠れ状態
        self.f=f
        self.f_difftmp=f_difftmp
        self.mu_record=np.array(self.mu)
    def output(self):
        return self.f(self.mu)+np.random.randn()*np.sqrt(self.var[1])

class recognizer(enviroment):
    def __init__(self) -> None:
        super().__init__()
        
    def f_diff(self,x):
        if self.f_difftmp:
            return self.f_difftmp(x)
        else:
            return (self.f(x+0.001)-self.f(x))/0.001
    def inference(self,y):
        self.mu=self.mu+((y-f(self.mu))*self.f_diff(self.mu)/self.var[0]+(self.mu0-self.mu)/self.var[1])*self.h
        self.mu_record=np.append(self.mu_record,self.mu)
env=enviroment()
env.mu=50
rec=recognizer()

n=10000
for i in range(n-1):
    y=env.output()
    rec.inference(y)
plt.plot(range(n),[env.mu for i in range(n)])
plt.plot(range(n),rec.mu_record)


plt.show()
