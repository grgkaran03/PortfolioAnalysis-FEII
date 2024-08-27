import math, numpy as np, matplotlib.pyplot as plt

def lookback_model(case, S0=100, T=1, K=100, r=0.08, sigma=0.3, M=10):
      
    delta= T/M
    R=math.exp(r*delta)
    if(case!=0):
        up_factor = math.exp(sigma*math.sqrt(delta) + (r - 0.5*sigma*sigma)*delta)
        down_factor = math.exp(-sigma*math.sqrt(delta) + (r - 0.5*sigma*sigma)*delta)
    else:
        up_factor = math.exp(sigma*math.sqrt(delta) )
        down_factor = math.exp(-sigma*math.sqrt(delta))
        
    probab= (R-down_factor)/(up_factor-down_factor)
    
    if(R<=down_factor or R>=up_factor):
        print(f"Arbitrage oppurtunity exist for M={M}")
        return -1,-1  
    
    def call_price(stepno,current,maxima):
        
        if(stepno==M):
            
            return max(0,maxima-K)
        
        return (probab*(call_price(stepno+1,current*up_factor,max(maxima,current*up_factor))) + (1-probab)*(call_price(stepno+1,current*down_factor,maxima)))/R
    
    def put_price(stepno,current,minima):
        if(stepno==M):
            
            return max(0,K-minima)
        
        return (probab*(put_price(stepno+1,current*up_factor,minima)) + (1-probab)*(put_price(stepno+1,current*down_factor,min(minima,current*down_factor))))/R
    
    return call_price(0,S0,S0),put_price(0,S0,S0)

def graph(x, y, yy, var, case):
    plt.plot(x,y, label='call option')
    plt.plot(x, yy, label='put option')
    plt.title(f'Set = {case+1}')
    plt.legend()
    plt.xlabel(var)
    plt.ylabel('Initial Option Price')
    plt.show()

for case in [0,1]:
    print(f'Set = {case}')
    for option in ['call', 'put']:
        print(f'initial {option} option value: {lookback_model(case=case)}')

for case in [0,1]:
    x=[]
    y=[]
    yy=[]
    for S0 in np.linspace(50, 150, 100):
        x.append(S0)
        c,p=lookback_model(case=case,S0=S0)
        y.append(c)
        yy.append(p)
    graph(x,y,yy,'S0', case)
    x.clear()
    y.clear()
    yy.clear()
    for K in np.linspace(50, 150, 100):
        x.append(K)
        c,p=lookback_model(case=case,K=K)
        y.append(c)
        yy.append(p)
    graph(x,y,yy,'K', case)
    x.clear()
    y.clear()
    yy.clear()
    for r in np.linspace(0.07, 0.09, 100):
        x.append(r)
        c,p=lookback_model(case=case,r=r)
        y.append(c)
        yy.append(p)
    graph(x,y,yy,'r', case)
    x.clear()
    y.clear()
    yy.clear()
    for sigma in np.linspace(0.1, 0.5, 100):
        x.append(sigma)
        c,p=lookback_model(case=case,sigma=sigma)
        y.append(c)
        yy.append(p)
    graph(x,y,yy,'sigma', case)
    x.clear()
    y.clear()
    yy.clear()
    for option in ['call', 'put']:
        for K in range(95, 106, 5):
            for M in range(3, 22):
                x.append(M)
                c,p=lookback_model(case=case,M=int(M), K=K)
                if(option=='call'):
                    y.append(c)
                else:
                    y.append(p)
            plt.plot(x,y)
            plt.title(f'Set = {case+1},option = {option}, K={K}')
            plt.xlabel('M')
            plt.ylabel('Initial Option Price')
            plt.show()
            x.clear()
            y.clear()
