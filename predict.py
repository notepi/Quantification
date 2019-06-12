#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:43:59 2019

@author: pan
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import tushare as ts
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.ticker as ticker
# from sklearn.externals import joblib
import datetime
import joblib

def getstockprice(stocklist):
    tt=[]
    tt.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    df=ts.get_realtime_quotes(stocklist) #Single stock symbol
    df=df[['name','price','time']]
    df['price']=df['price'].apply(float)
    temp=df['price'].tolist()
    # 对上证指数进行缩放
    temp[-2]=temp[-2]/200
    data=pd.DataFrame(data=temp).T
    data.columns=df['name']
    
    data.index=tt
    return data
 
    
if __name__ == "__main__":
    regtime=6
    mergenums=10
    stophour=16
    ts.set_token('4e980c06a141622965e416b016f45027fb5f0fa47f0d6d8863d4bd31')
    
    pro = ts.pro_api()
    '''
    数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，
    其中1min表示1分钟（类推1/5/15/30/60分钟） ，
    默认D。目前有120积分的用户自动具备试用权限（每分钟5次），正式权限请在QQ群私信群主。
    '''
    stock_name=['温氏股份', '益生股份', '仙坛股份', '圣农发展', '民和股份', '正邦科技', '天邦股份',
       '牧原股份', '上证指数','华英农业']
    stock_code=['300498','002458','002746','002299','002234',
               '002157','002124','002714','sh','002321']
    
    result_tmp=[]
    i=0
    tag_final=0
    try:
        final=pd.read_csv("./temp/final.csv",index_col=0)
    except:
        final=getstockprice(stock_code)
        pass
    linerecordtime=time.time()
    while(1):
        # 读取当前时间
        localtime=time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        # 读取当前小时
        hour=int(localtime[11:13])
        mines=int(localtime[14:16])
        mytime=hour*100+mines
        if mytime<930 or mytime>1500 or (mytime>1130 and mytime<1300):
            time.sleep(1)
            print("sleeping")
            time.sleep(1)
            continue
            pass

        result_tmp.append(getstockprice(stock_code))
        i=i+1
        time.sleep(1)
        if i >= mergenums:
            #间隔dealynums秒进行一次合并
            #重新归零
            i=0
            # 合并产生的result
            result_tmp=pd.concat(result_tmp) 
            final=pd.concat([final,result_tmp])  
            result_tmp=[]
        pass

        #拟合一次曲线
        if (time.time()-linerecordtime)>=regtime:
            # print((time.time()-linerecordtime))
            # print("=====")
            #时间重新赋值
            linerecordtime=time.time()
            #读取模型
            model = joblib.load("./model/stable__Model.pkl")
            # 强行合数据
            try:
                # 合并产生的result
                result_tmp=pd.concat(result_tmp) 
                final=pd.concat([final,result_tmp])  
                result_tmp=[]
                i=0
                pass
            except:
                pass
            day=localtime.split(" ")[0]
            a=final[day+' 09:20:00':day+' 11:30:00']
            b=final[day+' 13:00:00':day+' 15:00:00']
            final=pd.concat([a,b])
            # 不到时间不操作
            if len(final) >0 :
                yhat=model.predict(final.iloc[:,:-1])

                # tick_spacing=int(len(final)/30)+1
                # y=final.iloc[:,-1].values
                # ret=yhat-y
                # _, ax = plt.subplots(1,1)
                # ax.plot(final.iloc[:,-1].index,yhat,label='y_hat',color='r')
                # ax.plot(final.iloc[:,-1].index,y,label='HY',color='b')
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                # plt.xticks(rotation=45,size = 5)
                # plt.legend()
                
                # ax2 = ax.twinx()
                # _, =ax2.plot(ret, color='g',label='y_ret') # green
                # ax2.set_ylabel('ret', color='g')   
                # ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                
                # plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # plt.grid()
                # plt.show()
                final.to_csv("./temp/final.csv")
                mytemp=final.copy()
                mytemp['yhat']=yhat
                mytemp.to_csv("./temp/today.csv")
                pass
            pass

        model = joblib.load("./model/stable__Model.pkl") 
        tmp=getstockprice(stock_code)
        yhat=model.predict(tmp.iloc[:,:-1])
        y=tmp.iloc[:,-1]
    

        print("时间:%s|误差:%f|预测值:%f|实际值:%f"%\
             (time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()),(yhat-y)[0],yhat[0],y[0]),flush=True)



        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # print("误差:",(yhat-y)[0])
        # print('预测值:',yhat[0])
        # print('实际值:',y[0])

        pass #while结束


        #进行预测 













