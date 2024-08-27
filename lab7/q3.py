from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import math

def bsm(s,t,T,k,r,sigma):
    if t==T:
        return max(0,s-k),max(k-s,0)
    tau=T-t
    D_plus=(math.log(s/k)+(r+(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    D_minus=(math.log(s/k)+(r-(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    call=s*norm.cdf(D_plus)-(k*math.exp(-r*(tau))*norm.cdf(D_minus))
    put=(k*math.exp(-r*(tau))*norm.cdf(-1*D_minus))-s*norm.cdf(-1*D_plus)
    return call,put

time=np.linspace(0,1,100)
stock=np.linspace(0.0001,2,100)

time,stock=np.meshgrid(time,stock)
call=[]
put=[]

for i in range(len(time)):
    cp=[]
    pp=[]
    for j in range(len(stock)):
        c,p=bsm(stock[i][j],time[i][j],1,1,0.05,0.6)
        cp.append(c)
        pp.append(p)
    call.append(cp)
    put.append(pp)
call=np.array(call)
put=np.array(put)

ax=plt.axes(projection="3d")
ax.plot_surface(time,stock,call,cmap="viridis")
ax.set_xlabel("time(t)")
ax.set_ylabel("Stock price(s)")
ax.set_zlabel("Call Price")
plt.title("Variation of C(t,S(t)) for change in t and S(t)")
plt.show()

ax=plt.axes(projection="3d")
ax.plot_surface(stock,time,put,cmap="viridis")
ax.set_ylabel("time(t)")
ax.set_xlabel("Stock price(s)")
ax.set_zlabel("Put Price")
plt.title("Variation of P(t,S(t)) for change in t and S(t)")
plt.show()