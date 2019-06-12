import numpy as np
import pandas as pd
import os 
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
if __name__ == "__main__":
    while(1):
        data=pd.read_csv("./temp/today.csv")

        tick_spacing=int(len(data)/30)+1
        y=data["华英农业"]
        yhat=data["yhat"]
        ret=data["yhat"]-data["华英农业"]
        _, ax = plt.subplots(1,1)
        ax.plot(data.iloc[:,-1].index,yhat,label='y_hat',color='r')
        ax.plot(data.iloc[:,-1].index,y,label='HY',color='b')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(rotation=45,size = 5)
        plt.legend()
        
        ax2 = ax.twinx()
        _, =ax2.plot(ret, color='g',label='y_ret') # green
        ax2.set_ylabel('ret', color='g')   
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        
        plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        plt.grid()
        plt.show()
        time.sleep(10)
    pass