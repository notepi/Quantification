#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:43:59 2019

@author: pan
"""

import os
import time

# from sklearn.externals import joblib
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tushare as ts
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import r2_score  # R square
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)


# import pickle
def f(train,test,na):
    global tagname
    #训练数据
    y=train[tagname]
    X=train
    del X[tagname]
    
    #测试数据
    TagTest=test[tagname]
    XdataTest=test
    del XdataTest[tagname]
    
    # 预测
    model = Ridge()

    alpha_can = np.logspace(-3, 2, 10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(X, y) 
    # print(X.columns)
    
    # 保存模型
    joblib.dump(lasso_model, "./model/"+na+"_Model.pkl")
    print("./model/"+na+"_Model.pkl")
    ###########################################################################
    #测试数据
    data=XdataTest
    tag=TagTest
    y_hat = lasso_model.predict(data)
    #调用
    pca_mse=mean_squared_error(y_hat,tag)
    pca_mbe=mean_absolute_error(y_hat,tag)
    pca_r2=r2_score(y_hat,tag)

    
    #保存测试数据
    result=pd.DataFrame([tag.values,y_hat]).T
    result.columns=[tagname,'yhat']
    result.index=tag.index
    result["y_ret"]=result["yhat"]-result[tagname]
    result=result.sort_index(ascending=False)
    result.to_excel('./temp/'+na+"test.xlsx")   
    
    result=result.sort_index(ascending=True)
    tick_spacing = int(len(result)/60)+1

    #通过修改tick_spacing的值可以修改x轴的密度
    #1的时候1到16，5的时候只显示几个
    _, ax = plt.subplots(1,1)
    ax.plot(result.index,result['yhat'],label='y_hat',color='r')
    ax.plot(result.index,result[tagname],label='Yreal',color='b')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) 
    plt.xticks(rotation=45,size = 8)
    plt.legend()

#    plt.legend()
    
    ax2 = ax.twinx()
    _, =ax2.plot(result['y_ret'], color='g',label='y_ret') # green
    ax2.set_ylabel('ret', color='g')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    
    plt.title(na+'test')
    plt.grid()
    plt.savefig('./picture/'+na+'test.jpg')
    plt.show()    

    ###########################################################################
    #训练数据
    Tdata=X.copy()
    Tdata[tagname]=y
    _,test  = train_test_split(Tdata, train_size=0.99, random_state=1)
    tag=test[tagname]
    data=test.iloc[:,:-1]
    y_hat = lasso_model.predict(data)
    #调用

    pca_mse=mean_squared_error(y_hat,tag)
    pca_mbe=mean_absolute_error(y_hat,tag)
    pca_r2=r2_score(y_hat,tag)
    print("pca_mse",pca_mse)
    print("pca_mbe",pca_mbe)
    print("pca_r2",pca_r2)   
    
    
    #保存测试数据
    result=pd.DataFrame([tag.values,y_hat]).T
    result.columns=[tagname,'yhat']
    result.index=tag.index
    result["y_ret"]=result["yhat"]-result[tagname]
    result=result.sort_index(ascending=False)
    result.to_excel('./temp/'+na+"train.xlsx")   
    
    
    
    result=result.sort_index(ascending=True)
    tick_spacing = int(len(result)/60)+1
    #通过修改tick_spacing的值可以修改x轴的密度
    #1的时候1到16，5的时候只显示几个
    _, ax = plt.subplots(1,1)
    ax.plot(result.index,result['yhat'],label='y_hat',color='r')
    ax.plot(result.index,result[tagname],label='Yreal',color='b')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) 
    plt.xticks(rotation=45,size = 8)
    plt.legend()

#    plt.legend()
    
    ax2 = ax.twinx()
    _, =ax2.plot(result['y_ret'], color='g',label='y_ret') # green
    ax2.set_ylabel('ret', color='g')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    
    plt.title(na+'train')
    plt.grid()
    plt.savefig('./picture/'+na+'train.jpg')
    plt.show()
    
    
    return result,lasso_model


 
    
if __name__ == "__main__":

    ts.set_token('4e980c06a141622965e416b016f45027fb5f0fa47f0d6d8863d4bd31')
    
    pro = ts.pro_api()

    #读取股票参数
    '''
    数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，
    其中1min表示1分钟（类推1/5/15/30/60分钟） ，
    默认D。目前有120积分的用户自动具备试用权限（每分钟5次），正式权限请在QQ群私信群主。
    '''   
    # stock_code=['000876.sz','300498.sz','002458.sz','002746.sz','002299.sz',\
    #             '002234.sz','002321.sz','002157.sz','002124.sz',\
    #             '002714.sz','000001.SH']
    # stock_name=["新希望","温氏股份","益生股份",'仙坛股份','圣农发展','民和股份',\
    #             '华英农业','正邦科技','天邦股份','牧原股份','上证指数']
    print("========================================================")
    print("init")
    stockdata=pd.read_excel("./doc/stock.xlsx",dtype='str')
    stock_code=list(stockdata['stock_code'].apply(lambda x:x+'.')+stockdata['local'])
    stock_name=list(stockdata['stock_name'])
    tagname=stock_name[-1]

    startime_read=list(stockdata['startime'])[0]
    #截止日期要比真实的大1
    temp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    temp=str(int("".join(temp.split(" ")[0].split('-')))+1)
    endtime_read=temp
    print("start time is:",startime_read)
    print("end time is:",endtime_read)


    result=[]
    for name_index,stock in enumerate(stock_code):
        startime=startime_read
        endtime=endtime_read
        print(stock_name[name_index])
        #读取文件
        #有文件
        flags=1
        try:
            data=pd.read_csv("./data/"+stock_name[name_index]+".csv")
            data=data.sort_values(by='time',ascending=False)
            print('读取文件成功，重置读取起始时间')
            startime=str(int("".join(data["time"].values[0].split()[0].split("-"))) + 1)
        except:
            print('无文件，从头开始读取')
            #设置空表
            data=pd.DataFrame(columns=["time",stock_name[name_index]])
            pass

        print("get data from servers")
        while(1):
            # 一分钟最多5次，也可避免最开始全是空有异常
            print("sleep")
            time.sleep(16)
            print("start time is:",startime)
            print("end time is:",endtime)

            df = ts.pro_bar(ts_code=stock, start_date=startime, end_date=endtime,freq="1min")
            if len(df)<=0:
                print("更新完成")
                break
                pass
            # 读取到新文件
            # 获取第一列和第二列:时间和价格
            temp=df.iloc[:,1:3]
            temp.columns=["time",stock_name[name_index]]
            # 对数据进行融合
            data=pd.concat([data,temp])  
            #更新下一轮读取起始时间，既上一轮的结尾日期+1
            nexttime=int("".join(df["trade_time"].loc[1].split()[0].split("-")))+1
            startime=str(nexttime)
            print(len(df))
            print(nexttime)
            pass#while结束

        data=data.sort_values(by='time',ascending=False)
        #更新本地文件
        print("文件更新成功")
        data.to_csv("./data/"+stock_name[name_index]+".csv",index=False)
        result.append(data)
        pass#for结束
    #时间栏换成index，最后统一处理
    for j,i in enumerate(result):
        i.index=i['time']
        del i['time']
        result[j]=i
        pass
    # 合并最后读取的完整数据
    result=pd.concat(result,axis=1,sort=True)
    
    # finaldata=result.sort_values(by='time',ascending=False)
    finaldata=result.sort_index(ascending=True)

    # 空缺
    for i in list(range(finaldata.shape[-1])):
        finaldata=finaldata[finaldata.iloc[:,i].isna()==False]
        pass    

    y=finaldata[tagname]
    del finaldata[tagname]

    finaldata.insert(loc=finaldata.shape[1],column=tagname,value=y)


    
    finaldata['上证指数']=finaldata['上证指数']/200

    ###########################################################################
    # 前期全拟合，后期未拟合
    #模型
    print("--")
    train=finaldata.iloc[:-241,:]
    test=finaldata.iloc[-241:,:]
    re,model_or=f(train,test,na='Yesterday')
    print("--")
    
    # # 前期全拟合，后期部分未拟合
    # train=finaldata.iloc[:-1200,:]
    # test=finaldata.iloc[-1200:,:]
    # trainT,testT  = train_test_split(test, train_size=0.99, random_state=1)
    # train=pd.concat([train,trainT])
    # test=testT
    # re,model=f(train,test,na='b_')
    # model.best_estimator_.alpha

    
##########################################################################################    
    # 抽样拟合，用于更新模型
    train,test  = train_test_split(finaldata, train_size=0.99, random_state=1)
    re,model=f(train,test,na='stable_')
    
    #检验结果
    yhat=model.predict(finaldata.iloc[-241:,:-1])
    y=finaldata.iloc[-241:,-1].values
    ret=yhat-y
    y_or=model_or.predict(finaldata.iloc[-241:,:-1])
    
    tick_spacing = int(len(result)/60)+1
    _, ax = plt.subplots(1,1)
    ax.plot(finaldata.iloc[-241:,-1].index,yhat,label='y_hat',color='r')
    ax.plot(finaldata.iloc[-241:,-1].index,y,label='Yreal',color='b')
    ax.plot(finaldata.iloc[-241:,-1].index,y_or,label='y_or',color='y')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) 
    plt.xticks(rotation=270,size = 8)
    plt.legend()

    
    ax2 = ax.twinx()
    _, =ax2.plot(ret, color='g',label='y_ret') # green
    ax2.set_ylabel('ret', color='g')   
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    plt.title('check')
    plt.grid()
    plt.savefig('./picture/check.jpg')
    plt.show()
    
    pass
# '''