from scipy.stats import norm
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

call,put=bsm(1,0,1,1,0.05,0.6)
print('price of call option=', call)
print('price of put option=', put)