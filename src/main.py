import numpy as np
import matplotlib.pyplot as plt
import time

def f(x):
    return x**2
def f_diff(x):
    return 2*x
class enviroment():
    def __init__(self,x0=10,var=[0.1,0.1],h=0.0001,mu0=0,f=f,f_difftmp=f_diff) -> None:
        self.var=var# [x_var,y_var]
        self.h=h#　更新幅
        self.mu0=mu0#事前分布の平均
        self.mu=x0#mu0+np.random.randn()*np.sqrt(self.var[0])# 隠れ状態
        self.f=f
        self.f_difftmp=f_difftmp
        self.mu_record=np.array(self.mu)
    def output(self):
        return self.f(self.mu)+np.random.randn()*np.sqrt(self.var[1])

class recognizer(enviroment):
    def __init__(self,x0=10,analysis_method=1,var=[0.1,0.1],h=0.0001,mu0=9,f=f,f_difftmp=f_diff) -> None:
        super().__init__(x0=x0,var=var,h=h,mu0=mu0,f=f,f_difftmp=f_difftmp)
        self.analysis_method=analysis_method
        
    def f_diff(self,x):
        if self.f_difftmp:
            return self.f_difftmp(x)
        else:
            return (self.f(x+0.001)-self.f(x))/0.001
    def differential(self,x,y):
        return ((y-f(x))*self.f_diff(x)/self.var[0]+(self.mu0-x)/self.var[1])
    def inference(self,y):
        if self.analysis_method==0:#オイラー法
            self.mu+=self.differential(self.mu,y)*self.h
        elif self.analysis_method==1:#ルンゲクッタ法
            k1 = self.differential(self.mu, y)
            k2 = self.differential(self.mu + self.h * k1 / 2, y)
            k3 = self.differential(self.mu + self.h * k2 / 2, y)
            k4 = self.differential(self.mu + self.h * k3, y)

            self.mu += self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.mu_record=np.append(self.mu_record,self.mu)

h=0.00001
analysis_method=1
var=[np.e**-10,np.e**-16]
var=[0.01,0.01]
env=enviroment(var=var,x0=10,h=h)
rec=recognizer(var=var,x0=9,h=h,analysis_method=analysis_method)


n=int(0.001/h)
start=time.time()
for i in range(n-1):
    y=env.output()
    rec.inference(y)
    if i%10000==0:
        print(i)
print(time.time()-start)
plt.plot(range(n),[env.mu for i in range(n)])
plt.plot(range(n),rec.mu_record)


plt.show()
