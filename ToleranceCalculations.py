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
def stddev(arr):
    return np.std(arr)

def main(input_args):
    output_list = list()
    output_dict=dict()

    df =pd.read_csv("data/BenchmarkVariance.csv")
    df=df[1:100]
    stddevlist=list()
    for i in range(100):
        std=df[i:i+60]['Price']
        stddevlist.append(stddev(std))
        
    df['Standard Deviation']=pd.Series(stddevlist)
    df.head(107)
    df['DOD_Underlying']=df['Price'].diff()
    df['Lower Tolerance Range']=-(df['Standard Deviation'])
    df['Upper Tolerance Range']=(df['Standard Deviation'])
    df=df[['Price', 'DOD_Underlying','Lower Tolerance Range','Upper Tolerance Range']]

    graphList=list()
    output,plot1=plot_vol_surface(df)
    graphList.append(plot1)

    
    output_dict['title'] = "Tolerance"
    output_dict["output"] = df.iloc[0:10,:]#[:noOfRows]
    output_dict["graph"] = graphList
    output_dict["label"] ="Tolerance"   
    output_list.append(output_dict)
    return output_list


def plot_vol_surface(df):
    plt.figure(figsize=[10,10])
    plt.grid(True)
    plt.plot(df['Lower Tolerance Range'],label='Lower Tolerance Range')
    plt.plot(df['Upper Tolerance Range'],label='Upper Tolerance Range')
    plt.plot(df['DOD_Underlying'],label='Day Over Day Change')
    plt.legend(loc='upper left')
    valDate = datetime.strptime('01/01/2009', '%d/%m/%Y')
    output = "MACD=" * 80
    output = output + "\n"
    output = output + "Valuation date: " + str(valDate) + "\n"
    output_plot = "output" + os.path.sep + str(valDate) +'_surf1.png'
    plt.savefig(output_plot, transparent=True, dpi=500)    
    return  output,output_plot

if __name__ == '__main__':
    main()
