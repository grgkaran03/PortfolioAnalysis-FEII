import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def n_step_pricing(fin_pricing, M, p, q, rate):
	M=int(M)
	if(M<=0):
		return fin_pricing[0]	
	fin_pricing_step = []		
	for i in range(len(fin_pricing)-1):
		fin_pricing_step.append((p*fin_pricing[i] + q*fin_pricing[i+1])/rate)	
	return n_step_pricing(fin_pricing_step, M-1, p, q, rate)

def prices_1(S0=100,K=100,T=1,M=100,r=0.08,sigma=0.3,flag=1):
	M=int(M)
	if(flag==1):
		u = np.exp(sigma*np.sqrt(T/M) + (r-sigma*sigma/2)*T/M)
		d = np.exp(-sigma*np.sqrt(T/M) + (r-sigma*sigma/2)*T/M)
	else:
		u = np.exp(sigma*np.sqrt(T/M))
		d = np.exp(-sigma*np.sqrt(T/M))	
	rate = np.exp(r*T/M)	
	p = (rate-d)/(u-d)
	q = 1-p	
	if(p<0 or p>1):
		print(values)
		print(u)
		print(d)
		print(rate)
		print(p)
		exit("Arbitrage exists")	
	stock_prices = []	
	for i in range(int(M)+1):
		stock_prices.append(S0*np.math.pow(u,M-i)*np.math.pow(d,i))	
	fin_pricing = []	
	for i in range(int(M)+1):
		fin_pricing.append(max(stock_prices[i]-K, 0))
	
	call_price = n_step_pricing(fin_pricing, M, p, q, rate)	
	put_price = call_price + K*np.exp(-r*T) - S0	
	return call_price, put_price


bound = {'S0_low': 75, 'S0_up': 150, 'K_low': 75, 'K_up': 150, 'r_low': 0.01, 'r_up': 0.1, 'sigma_low': 0.1, 'sigma_up': 1, 'M_low': 50, 'M_up': 150}
parameters=['S0','K','r','sigma','M']

for flags in [0,1]:	
    for i in range(5):
        for j in range(i,5):
            if((i!=0 or j!=1) and (i!=2 or j!=3) and (i!=j)):
                continue

            values = {"S0":100,"K":100,"T":1,"M":100,"r":0.08,"sigma":0.3,"flag":flags}
            if(i!=j):
                param1=parameters[i]
                param2=parameters[j]
            
                x_range = np.linspace(bound[param1+'_low'],bound[param1+'_up'],20)
                y_range = np.linspace(bound[param2+'_low'],bound[param2+'_up'],20)
                x=[]
                y=[]
                call_prices=[]
                put_prices=[]
                for x_ in x_range:
                    for y_ in y_range:
                        x.append(x_)
                        y.append(y_)
                        values[param1]=x_
                        values[param2]=y_
                        call,put=prices_1(values['S0'],values['K'],values['T'],values['M'],values['r'],values['sigma'],values['flag'])
                        call_prices.append(call)
                        put_prices.append(put)
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter3D(x, y, call_prices)
                ax1.set_xlabel(f'{param1}')
                ax1.set_ylabel(f'{param2}')
                ax1.set_zlabel('Call Prices')
                ax1.set_title(f'Call Pricing')
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter3D(x, y, put_prices)
                ax2.set_xlabel(f'{param1}')
                ax2.set_ylabel(f'{param2}')
                ax2.set_zlabel('Put Prices')
                ax2.set_title(f'Put Pricing')
                plt.suptitle(f'Varying {param1} from {bound[param1+"_low"]} to {bound[param1+"_up"]} and {param2} from {bound[param2+"_low"]} to {bound[param2+"_up"]}')
                plt.show()
            else:
                if(i!=4):
                    param1=parameters[i]                
                    x_range = np.linspace(bound[param1+'_low'],bound[param1+'_up'],100)
                    x=[]
                    call_prices=[]
                    put_prices=[]
                    for x_ in x_range:
                        x.append(x_)
                        values[param1]=x_
                        call,put=prices_1(values['S0'],values['K'],values['T'],values['M'],values['r'],values['sigma'],values['flag'])
                        call_prices.append(call)
                        put_prices.append(put)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                    ax1.plot(x, call_prices, label='Call Prices', color='blue')
                    ax1.set_xlabel(param1)
                    ax1.set_ylabel('Prices')
                    ax1.set_title('Call Prices')
                    ax1.legend()
                    ax2.plot(x, put_prices, label='Put Prices', color='orange')
                    ax2.set_xlabel(param1)
                    ax2.set_ylabel('Prices')
                    ax2.set_title('Put Prices')
                    ax2.legend()
                    plt.suptitle(f'Varying {param1} from {bound[param1+"_low"]} to {bound[param1+"_up"]}')
                    plt.tight_layout()

                    plt.show()
                else:
                    for K__ in [95,100,105]:
                        param1=parameters[i]
                
                        x_range = np.linspace(bound[param1+'_low'],bound[param1+'_up'],100)
                        x=[]
                        call_prices=[]
                        put_prices=[]
                        for x_ in x_range:
                            x.append(x_)
                            values[param1]=x_
                            call,put=prices_1(values['S0'],K__,values['T'],values['M'],values['r'],values['sigma'],values['flag'])
                            call_prices.append(call)
                            put_prices.append(put)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                        ax1.plot(x, call_prices, label='Call Prices', color='blue')
                        ax1.set_xlabel(f'{param1} with K={K__}')
                        ax1.set_ylabel('Prices')
                        ax1.set_title('Call Prices')
                        ax1.legend()
                        ax2.plot(x, put_prices, label='Put Prices', color='orange')
                        ax2.set_xlabel(f'{param1} with K={K__}')
                        ax2.set_ylabel('Prices')
                        ax2.set_title('Put Prices')
                        ax2.legend()
                        plt.suptitle(f'Varying {param1} from {bound[param1+"_low"]} to {bound[param1+"_up"]}')
                        plt.tight_layout()
                        plt.show()