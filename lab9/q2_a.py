import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from datetime import datetime

def plot_3D(filename, opt_type):
  mpl.rcParams.update(mpl.rcParamsDefault)
  df = pd.read_csv(filename)
  df['Expiry'] = pd.to_datetime(df['Expiry'])
  df['Date'] = pd.to_datetime(df['Date'])

  strike, maturity, call, put = [], [], [], []
  for index, row in df.iterrows():
    num = random.random()
    if num <= 0.25:
      strike.append(row['Strike Price'])
      d1 = row['Expiry']
      d2 = row['Date']
      delta = d1 - d2
      maturity.append(delta.days)
      call.append(row['Call Price'])
      put.append(row['Put Price'])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(strike, maturity, call, marker = '.')
  ax.set_xlabel('Strike Price')
  ax.set_ylabel('Maturity (in days')
  ax.set_zlabel('Call Price')
  ax.set_title('3D plot for Call Option - {}'.format(opt_type))
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(strike, maturity, put, marker = '.')
  ax.set_xlabel('Strike Price')
  ax.set_ylabel('Maturity (in days')
  ax.set_zlabel('Put Price')
  ax.set_title('3D plot for Put Option - {}'.format(opt_type))
  plt.show()


def plot_2D(filename, opt_type):
  plt.rcParams["figure.figsize"] = (13, 4)
  df = pd.read_csv(filename)
  df['Expiry'] = pd.to_datetime(df['Expiry'])
  df['Date'] = pd.to_datetime(df['Date'])

  # Option price vs Strike price
  x, call, put = [], [], []
  for index, row in df.iterrows():
    num = random.random()
    if num <= 0.25:
      x.append(row['Strike Price'])
      call.append(row['Call Price'])
      put.append(row['Put Price'])

  plt.subplot(1, 2, 1)
  plt.scatter(x, call, marker = '.')
  plt.xlabel('Strike price')
  plt.ylabel('Call Price')
  plt.title('Call price vs Strike Price for {}'.format(opt_type))
  
  plt.subplot(1, 2, 2)
  plt.scatter(x, put, marker = '.')
  plt.xlabel('Strike price')
  plt.ylabel('Put Price')
  plt.title('Put price vs Strike Price for {}'.format(opt_type))
  plt.show()

  # Option price vs Maturity
  x, call, put = [], [], []
  random.seed(53)
  for index, row in df.iterrows():
    num = random.random()
    if num <= 0.05:
      d1 = row['Expiry']
      d2 = row['Date']
      delta = d1 - d2
      x.append(delta.days)
      call.append(row['Call Price'])
      put.append(row['Put Price'])

  plt.subplot(1, 2, 1)
  plt.scatter(x, call, marker = '.')
  plt.xlabel('Maturity (in days)')
  plt.ylabel('Call Price')
  plt.title('Call price vs Maturity for {}'.format(opt_type))
  
  plt.subplot(1, 2, 2)
  plt.scatter(x, put, marker = '.')
  plt.xlabel('Maturity (in days)')
  plt.ylabel('Put Price')
  plt.title('Put price vs Maturity for {}'.format(opt_type))
  plt.show()

files = ['NIFTYoptiondata', 'stockoptiondata_HEROMOTOCO', 'stockoptiondata_RELIANCE', 'stockoptiondata_TATAMOTORS', 'stockoptiondata_HDFCBANK', 'stockoptiondata_BAJAJ-AUTO']
opt_type = ['NSE Index', 'HEROMOTOCO.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'BAJAJ-AUTO.NS']

for idx in range(len(files)):
  files[idx] = './' + files[idx] + '.csv'
  plot_2D(files[idx], opt_type[idx])
  plot_3D(files[idx], opt_type[idx])