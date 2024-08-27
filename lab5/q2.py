import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_market_portfolio(filename):
    df = pd.read_csv(filename)
    # df.set_index('Date', inplace=True)
    daily_returns = (df['Open']-df['Close'])/df['Open']
    daily_returns = np.array(daily_returns)

    df = pd.DataFrame(np.transpose(daily_returns))

    M, sigma = np.mean(df, axis=0)*len(df)/5, df.std()

    mu_market = M[0]
    risk_market = sigma[0]

    print("Market return \t=", mu_market)
    print("Market risk \t=", risk_market*100, "%")

    return mu_market, risk_market



def compute_weights(m, C, mu):
    C_inv = np.linalg.inv(C)

    u = [1 for i in range(len(m))]

    p = [[1, u @ C_inv @ np.transpose(m)], [mu, m @ C_inv @ np.transpose(m)]]
    q = [[u @ C_inv @ np.transpose(u), 1], [m @ C_inv @ np.transpose(u), mu]]
    r = [[u @ C_inv @ np.transpose(u), u @ C_inv @ np.transpose(m)], [m @ C_inv @ np.transpose(u), m @ C_inv @ np.transpose(m)]]

    det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)
    det_p/=det_r
    det_q/=det_r

    w = det_p*(u@C_inv) + det_q*(m@C_inv)

    return w


def efficient_frontier(m, C, mu_rf):

    returns  = np.linspace(-2, 5, num=2000)

    u = np.array([1 for i in range(len(m))])
    risk = []


    for mu in returns:
        w = compute_weights(m, C, mu)
        sigma = np.sqrt(w@C@np.transpose(w))
        risk.append(sigma)

    weight_min_var = (u@np.linalg.inv(C))/(u@np.linalg.inv(C)@np.transpose(u))
    mu_min_var = weight_min_var @ np.transpose(m)
    risk_min_var = np.sqrt(weight_min_var @ C @ np.transpose(weight_min_var)) 


    returns_1, risks_1, returns_2, risks_2 = [],[],[],[]

    for i in range(len(returns)):
        if returns[i] >= mu_min_var: 
            returns_1.append(returns[i])
            risks_1.append(risk[i])
        else:
            returns_2.append(returns[i])
            risks_2.append(risk[i])

    

    # Market Portfolio!!
            
    market_portfolio_weights = (m - mu_rf*u)@np.linalg.inv(C)/((m - mu_rf*u)@np.linalg.inv(C)@np.transpose(u))
    mu_market = market_portfolio_weights @ np.transpose(m)
    risk_market = np.sqrt(market_portfolio_weights@C@np.transpose(market_portfolio_weights))

    plt.plot(risks_1, returns_1, color= 'blue', label='Efficient Frontier')
    plt.plot(risks_2, returns_2, color = 'red')
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve & Efficient Frontier")
    plt.plot(risk_market, mu_market, color = 'green', marker = 'o')
    plt.annotate('Market Portfolio (' + str(round(risk_market, 4)) + ', ' + str(round(mu_market, 4)) + ')', 
             xy=(risk_market, mu_market)) #xytext=(0.012, 0.8)

    plt.plot(risk_min_var, mu_min_var, color = 'green', marker = 'o')
    plt.annotate('Minimum Variance Portfolio (' + str(round(risk_min_var, 4)) + ', ' + str(round(mu_min_var, 4)) + ')', 
             xy=(risk_min_var, mu_min_var))#xytext=(risk_min_var, -0.6)
    
    plt.legend()
    plt.show()

    print("Market Portfolio Weights \t= ", market_portfolio_weights)
    print("Return \t\t\t\t= ", mu_market)
    print("Risk \t\t\t\t= ", risk_market * 100, " %")

    return mu_market, risk_market



def plot_CapitalMarketLine(m, C, mu_rf, mu_market, risk_market):
    returns = np.linspace(-2, 5, num = 2000)
    u = np.array([1 for i in range(len(m))])
    risk = []

    for mu in returns:
        w = compute_weights(m, C, mu)
        sigma = np.sqrt(w@C@np.transpose(w))
        risk.append(sigma)

    returns_cml = []

    risk_cml = np.linspace(0, 0.25, num = 2000)
    for i in risk_cml:
        returns_cml.append(mu_rf + (mu_market - mu_rf) * i / risk_market)
    
    slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf

    print("\nEquation of Capital Market Line is:")
    print("y = {:.4f} x + {:.4f}\n".format(slope, intercept))

    plt.plot(risk_market, mu_market, color = 'purple', marker = 'o')

    plt.annotate('Market Portfolio (' + str(round(risk_market, 4)) + ', ' + str(round(mu_market, 4)) + ')', 
             xy=(risk_market, mu_market)) #xytext=(0.012, 0.8)
    
    plt.plot(risk, returns, label = 'Minimum Variance Line')
    plt.plot(risk_cml, returns_cml, label = 'CML')
    plt.title("Capital Market Line with Minimum Variance Line")
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(risk_cml, returns_cml)
    plt.title("Capital Market Line")
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.show()


def plot_SecurityMarketLine(m, C, mu_rf, mu_market, mu_risk):
    beta_v = np.linspace(-1, 1, 2000)
    mu_v = mu_rf + (mu_market - mu_rf) * beta_v
    plt.plot(beta_v, mu_v)
    
    print("Equaiton SML:")
    print("mu = {:.2f} beta + {:.2f}".format(mu_market - mu_rf, mu_rf))

    plt.title('SML for all the 10 assets')
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()

def markets_plots(stocks_name, type, mu_market_index, risk_market_index, beta):
    daily_returns = []

    for i in range(len(stocks_name)):
        filename = 'Data\\' + type + '\\' + stocks_name[i] + '.csv'
        df = pd.read_csv(filename)
        # df.set_index('Date', inplace=True)
        df = df.pct_change()
        daily_returns.append(df['Open'])

    daily_returns = np.array(daily_returns)
    df = pd.DataFrame(np.transpose(daily_returns), columns = stocks_name)
    m = np.mean(df, axis = 0) * len(df) / 5
    C = df.cov()
    
    mu_market, risk_market = efficient_frontier(m, C, 0.05)
    plot_CapitalMarketLine(m, C, 0.05, mu_market, risk_market)

    if type == 'BSE' or type == 'NSE':
        plot_SecurityMarketLine(m, C, 0.05, mu_market_index, risk_market_index)
    else:
        plot_SecurityMarketLine(m, C, 0.05, mu_market, risk_market)


def compute_beta(stocks_name, main_filename, index_type):
    df = pd.read_csv(main_filename)
    # df.set_index('Date', inplace=True)
    daily_returns = (df['Open'] - df['Close'])/df['Open']

    daily_returns_stocks = []
        
    for i in range(len(stocks_name)):
        if index_type == 'NonNSE':
            filename = 'Data\\NonNSE\\' + stocks_name[i] + '.csv'
        elif index_type == 'NonBSE':
            filename = 'Data\\NonBSE\\' + stocks_name[i] + '.csv'
        else:
            filename = 'Data\\' + index_type[:3] + '\\' + stocks_name[i] + '.csv'
        df_stocks = pd.read_csv(filename)
        # df_stocks.set_index('Date', inplace=True)

        daily_returns_stocks.append((df_stocks['Open'] - df_stocks['Close'])/df_stocks['Open'])
        

    beta_values = []
    for i in range(len(stocks_name)):
        df_combined = pd.concat([daily_returns_stocks[i], daily_returns], axis = 1, keys = [stocks_name[i], index_type])
        C = df_combined.cov()

        beta = C[index_type][stocks_name[i]]/C[index_type][index_type]
        beta_values.append(beta)

    return beta_values




print("**********  Inference about stocks taken from BSE  **********")
stocks_name_BSE = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
          'INFY.BO', 'BAJFINANCE.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO']
beta_BSE = compute_beta(stocks_name_BSE, 'Data\BSE\^BSESN.csv', 'BSE Index')
mu_market_BSE, risk_market_BSE = find_market_portfolio('Data\BSE\^BSESN.csv')
markets_plots(stocks_name_BSE, 'BSE', mu_market_BSE, risk_market_BSE, beta_BSE)



print("\n\n**********  Inference about stocks taken from NSE  **********")
stocks_name_NSE = ['ACC.NS', 'GODREJIND.NS', 'HINDZINC.NS', 'IDEA.NS', 'IGL.NS',
          'LUPIN.NS', 'MAHABANK.NS', 'MGL.NS', 'PAGEIND.NS', 'TATACHEM.NS']
beta_NSE = compute_beta(stocks_name_NSE, 'Data\\NSE\\^NSEI.csv', 'NSE Index')
mu_market_NSE, risk_market_NSE = find_market_portfolio('Data\\NSE\\^NSEI.csv')
markets_plots(stocks_name_NSE, 'NSE', mu_market_NSE, risk_market_NSE, beta_NSE) 
  
  
print("\n\n**********  Inference about stocks not listed in BSE  with index taken from BSE values**********")
stocks_name_non_bse = ['BAJAJ-AUTO.NS', 'MPHASIS.NS', 'ONGC.NS', 'COLPAL.NS', 'SIEMENS.NS',
          'VEDL.NS', 'ZEEL.NS', 'VOLTAS.NS', 'TRENT.NS', 'TATAPOWER.NS']
beta_non_index_BSE = compute_beta(stocks_name_non_bse, 'Data\BSE\^BSESN.csv', 'NonBSE')
markets_plots(stocks_name_non_bse, 'NonBSE', mu_market_BSE, risk_market_BSE, beta_non_index_BSE) 


print("\n\n**********  Inference about stocks not taken from any index  with index taken from NSE values**********")
stocks_name_non_nse = ['HAVELLS.NS', 'HAL.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'AMBUJACEM.NS', 
           'IOC.NS', 'NAUKRI.NS', 'INDIGO.NS', 'JINDALSTEL.NS', 'BANKBARODA.NS']
beta_non_index_NSE = compute_beta(stocks_name_non_nse, 'Data\\NSE\\^NSEI.csv', 'NonNSE')
markets_plots(stocks_name_non_nse, 'NonNSE', mu_market_NSE, risk_market_NSE, beta_non_index_NSE) 



print("**********  Beta for securities in BSE  **********")
beta_BSE = compute_beta(stocks_name_BSE, 'Data\\BSE\\^BSESN.csv', 'BSE Index')

for i in range(len(beta_BSE)):
  print("{}\t\t=\t\t{}".format(stocks_name_BSE[i], beta_BSE[i]))



print("\n\n**********  Beta for securities in NSE  **********")
beta_NSE = compute_beta(stocks_name_NSE, 'Data\\NSE\\^NSEI.csv', 'NSE Index')

for i in range(len(beta_NSE)):
  print("{}\t\t=\t\t{}".format(stocks_name_NSE[i], beta_NSE[i]))
  
  

print("\n\n**********  Beta for securities in non-index using BSE Index  **********")
beta_non_BSE = compute_beta(stocks_name_non_bse, 'Data\\BSE\\^BSESN.csv', 'NonBSE')

for i in range(len(beta_non_BSE)):
  print("{}\t\t=\t\t{}".format(stocks_name_non_nse[i], beta_non_BSE[i]))
  
  


print("\n\n**********  Beta for securities in non-index using NSE Index  **********")
beta_non_NSE = compute_beta(stocks_name_non_nse, 'Data\\NSE\\^NSEI.csv', 'NonNSE')

for i in range(len(beta_non_NSE)):
  print("{}\t\t=\t\t{}".format(stocks_name_non_nse[i], beta_non_NSE[i]))
