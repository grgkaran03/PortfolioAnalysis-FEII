import numpy as np, matplotlib.pyplot as plt

def american_pricing(option,S0=100,K=100,T=1,M=100,r=0.08,sigma=0.3):

    delt=T/M
    u=np.exp(sigma*np.sqrt(delt)+(r-sigma*sigma/2)*delt)
    d=np.exp(-sigma*np.sqrt(delt)+(r-sigma*sigma/2)*delt)

    R=np.exp(r*delt)
    p=(R-d)/(u-d)
    q=1-p

    if(p>1 or p<0):
        exit("Arbitrage exits")
    dp=[[0 for i in range(M+1)] for j in range(M+1)]
    chk=[[False for i in range(M+1)] for j in range(M+1)]

    def pricing(step, plus):
        if chk[step][plus]:
            return dp[step][plus]
        chk[step][plus]=True

        if(step==M):
            dp[step][plus]= max(0, (-1 if option == 'put' else 1) *(-K+S0*(u**plus)*(d**(step-plus))))
        else:
            dp[step][plus]=max(0,(-1 if option=='put' else 1)*(-K+S0*(u**plus)*(d**(step-plus))),(p*pricing(step+1, plus+1)+q*pricing(step+1,plus))/R)
        return dp[step][plus]
    return pricing(0,0)

def graph(x, y, yy, var):
    plt.plot(x,y, label='call option')
    plt.plot(x, yy, label='put option')
    plt.legend()
    plt.xlabel(var)
    plt.ylabel('Initial Option Price')
    plt.show()

for option in ['call', 'put']:
    print(f'initial {option} option value: {american_pricing(option)}')

x=[]
y=[]
yy=[]
for S0 in np.linspace(50, 150, 100):
    x.append(S0)
    y.append(american_pricing('call', S0=S0))
    yy.append(american_pricing('put',  S0=S0))
graph(x,y,yy,'S0' )
x.clear()
y.clear()
yy.clear()
for K in np.linspace(50, 150, 100):
    x.append(K)
    y.append(american_pricing('call',  K=K))
    yy.append(american_pricing('put',  K=K))
graph(x,y,yy,'K')
x.clear()
y.clear()
yy.clear()
for r in np.linspace(0.07, 0.09, 100):
    x.append(r)
    y.append(american_pricing('call',  r=r))
    yy.append(american_pricing('put',  r=r))
graph(x,y,yy,'r')
x.clear()
y.clear()
yy.clear()
for sigma in np.linspace(0.1, 0.5, 100):
    x.append(sigma)
    y.append(american_pricing('call',  sigma=sigma))
    yy.append(american_pricing('put',  sigma=sigma))
graph(x,y,yy,'sigma')
x.clear()
y.clear()
yy.clear()
for option in ['call', 'put']:
    for K in [95, 100, 105]:
        for M in np.linspace(50, 150, 100):
            x.append(M)
            y.append(american_pricing(option,  M=int(M), K=K))
        plt.plot(x,y)
        plt.title(f'{option} option, K={K}')
        plt.xlabel('M')
        plt.ylabel('Initial Option Price')
        plt.show()
        x.clear()
        y.clear()

