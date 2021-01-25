from datetime import datetime
from datetime import timedelta
import QuantLib as quant
import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import os, sys
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

def invoke(input_args=None):    
    main(fx_input_args)
noOfRows    = 10
def main(input_args):
    output_list = list()
    output_dict=dict()
#     print("main of Moving Average")
    df =pd.read_csv("data/TIME_SERIES.csv")
    df['MAVG50'] = df.iloc[:,1].rolling(window=50).mean()
    df['MAVG100'] = df.iloc[:,1].rolling(window=100).mean()
    df['MAVG50-100']=df['MAVG50']-df['MAVG100']
    
    df['EWMA50']=df.iloc[:,1].ewm(span=50, adjust=True).mean()
    df['EWMA100']=df.iloc[:,1].ewm(span=100, adjust=True).mean()    
    df['MACD'] = df['EWMA50']-df['EWMA100']
    exp3 = df['MACD'].ewm(span=50, adjust=True).mean()
#     plt.plot(df.ds, macd, label='AMD MACD', color = '#EBD2BE')
#     plt.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')
#     plt.legend(loc='upper left')
#     plt.show()
    
    graphList=list()
    output,plot1=plot_vol_surface(df)
    graphList.append(plot1)
    output,plot2=plot_vol_surface_macd(df)
    graphList.append(plot2)
    
    output_dict['title'] = "Trading Signal"
    output_dict["output"] = df.iloc[50:150,:]#[:noOfRows]
    output_dict["graph"] = graphList
    output_dict["label"] ="Trading Signals of TimeSeries"   
    output_list.append(output_dict)
    return output_list


def plot_vol_surface_macd(df):
    plt.figure(figsize=[10,10])
    plt.grid(True)
    plt.plot(df['EWMA50'],label='EWMA-50')
    plt.plot(df['EWMA100'],label='EWMA-100')
    plt.plot(df['MACD'],label='MACD')
    plt.legend(loc='upper left')
    valDate = datetime.strptime('01/01/2009', '%d/%m/%Y')
    output = "MACD=" * 80
    output = output + "\n"
    output = output + "Valuation date: " + str(valDate) + "\n"
    output_plot = "output" + os.path.sep + str(valDate) +'_surf1.png'
    plt.savefig(output_plot, transparent=True, dpi=500)    
    return  output,output_plot

def plot_vol_surface(df):
    plt.figure(figsize=[10,10])
    plt.grid(True)
    plt.plot(df['Price'],label='Price')
    
    plt.plot(df['MAVG50'],label='MAVG-50')
    plt.plot(df['MAVG100'],label='MAVG-100')
    plt.plot(df['MAVG50-100'],label='MAVG50-100')
    plt.legend(loc='upper left')
    valDate = datetime.strptime('01/01/2009', '%d/%m/%Y')
    output = "MAVG=" * 80
    output = output + "\n"
    output = output + "Valuation date: " + str(valDate) + "\n"
    output_plot = "output" + os.path.sep + str(valDate) +'_surf.png'
    plt.savefig(output_plot, transparent=True, dpi=500)    
    return  output,output_plot

if __name__ == '__main__':
    main()
