from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import math

def plot3d(stock,time,price,zlabel,title):
    x=[]
    y=[]
    z=[]
    for xx in range(len(time)):
        for yy in range(len(stock)):
            y.append(yy)
            x.append(xx)
            z.append(price[xx][yy])
    ax=plt.axes(projection="3d")
    ax.scatter3D(x,y,z)
    plt.title(title)
    ax.set_ylabel("Stock Price(s)")
    ax.set_xlabel("Time(t)")
    ax.set_zlabel(zlabel)
    plt.show()

def bsm(s,t,T,k,r,sigma):
    if t==T:
        return max(0,s-k),max(k-s,0)
    tau=T-t
    D_plus=(math.log(s/k)+(r+(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    D_minus=(math.log(s/k)+(r-(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    call=s*norm.cdf(D_plus)-(k*math.exp(-r*(tau))*norm.cdf(D_minus))
    put=(k*math.exp(-r*(tau))*norm.cdf(-1*D_minus))-s*norm.cdf(-1*D_plus)
    return call,put

time=[0,0.2,0.4,0.6,0.8,1]
stock=np.arange(0.01,2.01,0.01)
call=[]
put=[]
for t in time:
    callprice=[]
    for s in stock:
        c,p=bsm(s,t,1,1,0.05,0.6)
        callprice.append(c)
    plt.plot(stock,callprice,label="t= {}".format(t))
    call.append(callprice)
plt.xlabel("stock price(s)")
plt.ylabel("Call price")
plt.legend()
plt.grid()
plt.title("Variation of C(t,S(t)) for change in S(t)")
plt.show()

for t in time:
    putprice=[]
    for s in stock:
        c,p=bsm(s,t,1,1,0.05,0.6)
        putprice.append(p)
    plt.plot(stock,putprice,label="t= {}".format(t))
    put.append(putprice)
plt.xlabel("stock price(s)")
plt.ylabel("Put price")
plt.legend()
plt.grid()
plt.title("Variation of P(t,S(t)) for change in S(t)")
plt.show()

plot3d(stock,time,call,"Call Price","Variation of C(t,S(t)) for change in t and S(t)")
plot3d(stock,time,put,"Put Price","Variation of P(t,S(t)) for change in t and S(t)")