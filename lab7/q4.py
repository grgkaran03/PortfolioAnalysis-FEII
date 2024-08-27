from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import math
from tabulate import tabulate as tab

stock=[0.4,0.6,0.8,1,1.2]
Time=np.arange(0.01,5.01,0.01)
strike=np.linspace(0.01,2,500)
returnrate=np.linspace(0,1,500,endpoint=False)
risk=np.linspace(0.001,1,500)
stocks=np.linspace(0.01,2,500)

def bsm(s,t,T,k,r,sigma):
    if t==T:
        return max(0,s-k),max(k-s,0)
    tau=T-t
    D_plus=(math.log(s/k)+(r+(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    D_minus=(math.log(s/k)+(r-(0.5*sigma*sigma))*(tau))/(sigma*math.sqrt(tau))
    call=s*norm.cdf(D_plus)-(k*math.exp(-r*(tau))*norm.cdf(D_minus))
    put=(k*math.exp(-r*(tau))*norm.cdf(-1*D_minus))-s*norm.cdf(-1*D_plus)
    return call,put

def func_T(t,k,r,sigma,xlabel):
    call=[]
    put=[]
    for s in stock:
        callprice=[]
        if s==0.8:
            entries=[]
            cnt=0
        for T in Time:
            c,p=bsm(s,t,T,k,r,sigma)
            callprice.append(c)
            if s==0.8:
                cnt+=1
                if cnt%50==0:
                    entries.append([T,c,p])
        plt.plot(Time,callprice,label="s= {}".format(s))
        call.append(callprice)
    plt.xlabel(xlabel)
    plt.ylabel("Call price")
    plt.legend()
    plt.grid()
    plt.title("Variation of C(t,S(t)) wrt T")
    plt.show()
    for s in stock:
        putprice=[]
        for T in Time:
            c,p=bsm(s,t,T,k,r,sigma)
            putprice.append(p)
        plt.plot(Time,putprice,label="s= {}".format(s))
        put.append(putprice)
    plt.xlabel(xlabel)
    plt.ylabel("Put price")
    plt.legend()
    plt.grid()
    plt.title("Variation of P(t,S(t)) wrt T")
    plt.show()
    table=tab(entries,[xlabel,"C(t,S(t))","P(t,S(t))"],tablefmt="grid")
    print(table)

def func_K(t,T,r,sigma,xlabel):
    call=[]
    put=[]
    for s in stock:
        callprice=[]
        if s==0.8:
            entries=[]
            cnt=0
        for k in strike:
            c,p=bsm(s,t,T,k,r,sigma)
            callprice.append(c)
            if s==0.8:
                cnt+=1
                if cnt%50==0:
                    entries.append([k,c,p])
        plt.plot(strike,callprice,label="s= {}".format(s))
        call.append(callprice)
    plt.xlabel(xlabel)
    plt.ylabel("Call price")
    plt.legend()
    plt.grid()
    plt.title("Variation of C(t,S(t)) wrt k")
    plt.show()
    for s in stock:
        putprice=[]
        for k in strike:
            c,p=bsm(s,t,T,k,r,sigma)
            putprice.append(p)
        plt.plot(strike,putprice,label="s= {}".format(s))
        put.append(putprice)
    plt.xlabel(xlabel)
    plt.ylabel("Put price")
    plt.legend()
    plt.grid()
    plt.title("Variation of P(t,S(t)) wrt k")
    plt.show()
    table=tab(entries,[xlabel,"C(t,S(t))","P(t,S(t))"],tablefmt="grid")
    print(table)

def func_r(t,T,k,sigma,xlabel):
    call=[]
    put=[]
    for s in stock:
        callprice=[]
        if s==0.8:
            entries=[]
            cnt=0
        for r in returnrate:
            c,p=bsm(s,t,T,k,r,sigma)
            callprice.append(c)
            if s==0.8:
                cnt+=1
                if cnt%50==0:
                    entries.append([r,c,p])
        plt.plot(returnrate,callprice,label="s= {}".format(s))
        call.append(callprice)
    plt.xlabel(xlabel)
    plt.ylabel("Call price")
    plt.legend()
    plt.grid()
    plt.title("Variation of C(t,S(t)) wrt r")
    plt.show()
    for s in stock:
        putprice=[]
        for r in returnrate:
            c,p=bsm(s,t,T,k,r,sigma)
            putprice.append(p)
        plt.plot(returnrate,putprice,label="s= {}".format(s))
        put.append(putprice)
    plt.xlabel(xlabel)
    plt.ylabel("Put price")
    plt.legend()
    plt.grid()
    plt.title("Variation of P(t,S(t)) wrt r")
    plt.show()
    table=tab(entries,[xlabel,"C(t,S(t))","P(t,S(t))"],tablefmt="grid")
    print(table)

def func_sigma(t,T,k,r,xlabel):
    call=[]
    put=[]
    for s in stock:
        callprice=[]
        if s==0.8:
            entries=[]
            cnt=0
        for sigma in risk:
            c,p=bsm(s,t,T,k,r,sigma)
            callprice.append(c)
            if s==0.8:
                cnt+=1
                if cnt%50==0:
                    entries.append([sigma,c,p])
        plt.plot(risk,callprice,label="s= {}".format(s))
        call.append(callprice)
    plt.xlabel(xlabel)
    plt.ylabel("Call price")
    plt.legend()
    plt.grid()
    plt.title("Variation of C(t,S(t)) wrt sigma")
    plt.show()
    for s in stock:
        putprice=[]
        for sigma in risk:
            c,p=bsm(s,t,T,k,r,sigma)
            putprice.append(p)
        plt.plot(risk,putprice,label="s= {}".format(s))
        put.append(putprice)
    plt.xlabel(xlabel)
    plt.ylabel("Put price")
    plt.legend()
    plt.grid()
    plt.title("Variation of P(t,S(t)) wrt sigma")
    plt.show()
    table=tab(entries,[xlabel,"C(t,S(t))","P(t,S(t))"],tablefmt="grid")
    print(table)

def func_K_and_r(s,t,T,sigma,xlabel,ylabel):
    strike1,returnrate1=np.meshgrid(strike,returnrate)
    callprice=[]
    putprice=[]
    for i in range(len(strike)):
        call=[]
        put=[]
        for j in range(len(returnrate)):
            c,p=bsm(s,t,T,strike1[i][j],returnrate1[i][j],sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,returnrate1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in k and r")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,returnrate1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in k and r")
    plt.show()

def func_K_and_sigma(s,t,T,r,xlabel,ylabel):
    strike1,risk1=np.meshgrid(strike,risk)
    callprice=[]
    putprice=[]
    for i in range(len(strike)):
        call=[]
        put=[]
        for j in range(len(risk)):
            c,p=bsm(s,t,T,strike1[i][j],r,risk1[i][j])
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,risk1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in k and sigma")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,risk1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in k and sigma")
    plt.show()

def func_K_and_T(s,t,r,sigma,xlabel,ylabel):
    strike1,Time1=np.meshgrid(strike,Time)
    callprice=[]
    putprice=[]
    for i in range(len(strike)):
        call=[]
        put=[]
        for j in range(len(Time)):
            c,p=bsm(s,t,Time1[i][j],strike1[i][j],r,sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,Time1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in k and T")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,Time1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in k and T")
    plt.show()

def func_K_and_s(t,T,r,sigma,xlabel,ylabel):
    strike1,stocks1=np.meshgrid(strike,stocks)
    callprice=[]
    putprice=[]
    for i in range(len(strike)):
        call=[]
        put=[]
        for j in range(len(stocks1)):
            c,p=bsm(stocks1[i][j],t,T,strike1[i][j],r,sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,stocks1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in k and S(t)")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(strike1,stocks1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in k and S(t)")
    plt.show()

def func_T_and_s(t,k,r,sigma,xlabel,ylabel):
    Time1,stocks1=np.meshgrid(Time,stocks)
    callprice=[]
    putprice=[]
    for i in range(len(Time)):
        call=[]
        put=[]
        for j in range(len(stocks)):
            c,p=bsm(stocks1[i][j],t,Time1[i][j],k,r,sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,stocks1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in T and S(t)")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,stocks1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in T and S(t)")
    plt.show()

def func_T_and_r(s,t,k,sigma,xlabel,ylabel):
    Time1,returnrate1=np.meshgrid(Time,returnrate)
    callprice=[]
    putprice=[]
    for i in range(len(Time)):
        call=[]
        put=[]
        for j in range(len(returnrate)):
            c,p=bsm(s,t,Time1[i][j],k,returnrate1[i][j],sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,returnrate1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in T and r")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,returnrate1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in T and r")
    plt.show()

def func_T_and_sigma(s,t,k,r,xlabel,ylabel):
    Time1,risk1=np.meshgrid(Time,risk)
    callprice=[]
    putprice=[]
    for i in range(len(Time)):
        call=[]
        put=[]
        for j in range(len(risk)):
            c,p=bsm(s,t,Time1[i][j],k,r,risk1[i][j])
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,risk1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in T and sigma")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(Time1,risk1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in T and sigma")
    plt.show()

def func_r_and_s(t,T,k,sigma,xlabel,ylabel):
    returnrate1,stocks1=np.meshgrid(returnrate,stocks)
    callprice=[]
    putprice=[]
    for i in range(len(returnrate)):
        call=[]
        put=[]
        for j in range(len(stocks)):
            c,p=bsm(stocks1[i][j],t,T,k,returnrate1[i][j],sigma)
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(returnrate1,stocks1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in r and S(t)")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(returnrate1,stocks1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in r and S(t)")
    plt.show()

def func_r_and_sigma(s,t,T,k,xlabel,ylabel):
    returnrate1,risk1=np.meshgrid(returnrate,risk)
    callprice=[]
    putprice=[]
    for i in range(len(returnrate)):
        call=[]
        put=[]
        for j in range(len(risk)):
            c,p=bsm(s,t,T,k,returnrate1[i][j],risk1[i][j])
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(returnrate1,risk1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in r and sigma")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(returnrate1,risk1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in r and sigma")
    plt.show()

def func_sigma_and_s(t,T,k,r,xlabel,ylabel):
    risk1,stocks1=np.meshgrid(risk,stocks)
    callprice=[]
    putprice=[]
    for i in range(len(risk)):
        call=[]
        put=[]
        for j in range(len(stocks)):
            c,p=bsm(stocks1[i][j],t,T,k,r,risk1[i][j])
            call.append(c)
            put.append(p)
        callprice.append(call)
        putprice.append(put)
    callprice=np.array(callprice)
    putprice=np.array(putprice)

    ax=plt.axes(projection="3d")
    ax.plot_surface(risk1,stocks1,callprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Call Price")
    plt.title("Variation of C(t,S(t)) for change in sigma and S(t)")
    plt.show()

    ax=plt.axes(projection="3d")
    ax.plot_surface(risk1,stocks1,putprice,cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Put Price")
    plt.title("Variation of P(t,S(t)) for change in sigma and S(t)")
    plt.show()

func_T(0,1,0.05,0.6,"T")
func_K(0,1,0.05,0.6,"k")
func_r(0,1,1,0.6,"r")
func_sigma(0,1,1,0.05,"sigma")
func_K_and_r(0.8,0,1,0.6,"K","r")
func_K_and_sigma(0.8,0,1,0.05,"K","sigma")
func_K_and_T(0.8,0,0.05,0.6,"K","T")
func_K_and_s(0,1,0.05,0.6,"K","S(t)")
func_T_and_s(0,1,0.05,0.6,"T","S(t)")
func_T_and_r(0.8,0,1,0.6,"T","r")
func_T_and_sigma(0.8,0,1,0.05,"T","sigma")
func_r_and_s(0,1,1,0.6,"r","S(t)")
func_r_and_sigma(0.8,0,1,1,"r","sigma")
func_sigma_and_s(0,1,1,0.05,"sigma","S(t)")