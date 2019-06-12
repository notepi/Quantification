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
from sklearn.externals import joblib

def f(train,test,na):
    #训练数据
    y=train[tagname]
    X=train.iloc[:,1:]
    del X[tagname]
    
    #测试数据
    TagTest=test[tagname]
    XdataTest=test.iloc[:,1:]
    del XdataTest[tagname]
    
    # 预测
    model = Ridge()

    alpha_can = np.logspace(-3, 2, 10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(X, y) 
    print(X.columns)
    
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
    
#    t = np.arange(len(tag))
#    plt.figure()
#    plt.plot(XdataTest.index, tag,'-',label='a')
#    plt.plot(XdataTest.index, y_hat,'-',label='b')
#    plt.title('function')
#    plt.legend()
#    plt.show()   
    
    #保存测试数据
    result=pd.DataFrame([tag.values,y_hat]).T
    result.columns=[tagname,'yhat']
    result.index=tag.index
    result["y_ret"]=result["yhat"]-result[tagname]
    result=result.sort_index(ascending=False)
    result.to_excel(na+"test.xlsx")   
    
    result=result.sort_index(ascending=True)
    tick_spacing = 10
    #通过修改tick_spacing的值可以修改x轴的密度
    #1的时候1到16，5的时候只显示几个
    _, ax = plt.subplots(1,1)
    ax.plot(result.index,result['yhat'],label='y_hat',color='r')
    ax.plot(result.index,result[tagname],label='HY',color='b')
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
    result.to_excel(na+"train.xlsx")   
    
    
    
    result=result.sort_index(ascending=True)
    tick_spacing = 3
    #通过修改tick_spacing的值可以修改x轴的密度
    #1的时候1到16，5的时候只显示几个
    _, ax = plt.subplots(1,1)
    ax.plot(result.index,result['yhat'],label='y_hat',color='r')
    ax.plot(result.index,result[tagname],label='HY',color='b')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) 
    plt.xticks(rotation=270,size = 8)
    plt.legend()

#    plt.legend()
    
    ax2 = ax.twinx()
    _, =ax2.plot(result['y_ret'], color='g',label='y_ret') # green
    ax2.set_ylabel('ret', color='g')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    
    plt.title(na+'train')
    plt.grid()
    plt.show()
    
    
    return result,lasso_model


 
    
if __name__ == "__main__":


    
    
#    ts.set_token('4e980c06a141622965e416b016f45027fb5f0fa47f0d6d8863d4bd31')
    
#    pro = ts.pro_api()
    '''
    数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，
    其中1min表示1分钟（类推1/5/15/30/60分钟） ，
    默认D。目前有120积分的用户自动具备试用权限（每分钟5次），正式权限请在QQ群私信群主。
    '''
    stock_code=['000876.sz','300498.sz','002458.sz','002746.sz','002299.sz',\
                '002234.sz','002321.sz','002157.sz','002124.sz',\
                '002714.sz','000001.SH']
    stock_name=["新希望","温氏股份","益生股份",'仙坛股份','圣农发展','民和股份',\
                '华英农业','正邦科技','天邦股份','牧原股份','上证指数']

    tagname="华英农业"
    result=[]
    for name_index,stock in enumerate(stock_code):
        # 最早从12月5日开始
#        startime="20181205"
        #截止日期要比真实的大1
#        endtime="20190607"
        startime="20190610"
        #截止日期要比真实的大1
        endtime="20190611"
        print(stock_name[name_index])
        # 防止无限次
        i=0
        
        
        #读取文件
        #有文件
        flags=1
        try:
            fdata=pd.read_csv("./data/"+stock_name[name_index]+".csv")
            fdata=fdata.sort_values(by='time',ascending=False)

            # 最新的时间，不需要读取
            if (int("".join(fdata["time"].values[0].split()[0].split("-"))) + 1 >= int(endtime)) :
                print('c')
                result.append(fdata)
                continue
            pass
        except:
            print('d')
            flags=0
            fdata=pd.DataFrame(columns=["time",stock_name[name_index]])
            pass

        
        #文件存在，不需要从头读取
        if flags:
            print('a')
            startime=str(int("".join(fdata["time"].values[0].split()[0].split("-"))) + 1)
            data=fdata
            pass
        else:
            print('b')
            data=pd.DataFrame(columns=["time",stock_name[name_index]])
            pass
        print("get data from servers")
        

        while(1):
            i=i+1
            df = ts.pro_bar(ts_code=stock, start_date=startime, end_date=endtime,freq="1min")
            # 获取第一列和第二列:时间和价格
            temp=df.iloc[:,1:3]
            temp.columns=["time",stock_name[name_index]]
            
            # 对数据进行融合
            data=pd.concat([data,temp])
            
            #下一轮
            # 获取最新一组数据的最后日期
            nexttime=int("".join(df["trade_time"].loc[1].split()[0].split("-")))
            startime=str(nexttime)
            #获取到了最新数据，则停止
            #截止日期要比真实的大1
            print(len(df))
            if (int(endtime)-1)<=nexttime:
                # 一分钟最多5次，避免最开始全是空有异常
                time.sleep(16)
                print("==")
                break
            print(nexttime)
            #最多500次，防止无限循环
            if i>500:
                #
                print(nexttime)
                break
                pass
            # 一分钟最多5次
            time.sleep(16)
            
            pass#while结束
        try:
            data=pd.concat([fdata,data])
            print("--")

        except:
            print("==")
            #无最新数据
            data=fdata
            pass
        data=data.sort_values(by='time',ascending=False)
        
        #更新本地文件
        data.to_csv("./data/"+stock_name[name_index]+".csv",index=False)
        result.append(data)

        pass#for结束
    print("==")
    for j,i in enumerate(result):
        i.index=i['time']
        del i['time']
        result[j]=i
        pass

    result=pd.concat(result,axis=1)
    
    finaldata=result.sort_values(by='time',ascending=False)
    finaldata=finaldata.sort_index(ascending=True)
    
    y=finaldata[tagname]
    del finaldata[tagname]
    finaldata.insert(loc=finaldata.shape[1],column=tagname,value=y)
    # 空缺
    for i in list(range(10)):
        finaldata=finaldata[finaldata.iloc[:,i].isna()==False]
        pass
    
    finaldata['上证指数']=finaldata['上证指数']/200


    ###########################################################################
    # 前期全拟合，后期未拟合
    #模型
    print("--")
    train=finaldata.iloc[:-241,:]
    test=finaldata.iloc[-241:,:]
    re=f(train,test,na='A_')
    print("--")
    
#    # 前期全拟合，后期部分未拟合
#    train=finaldata.iloc[:-1200,:]
#    test=finaldata.iloc[-1200:,:]
#    trainT,testT  = train_test_split(test, train_size=0.99, random_state=1)
#    train=pd.concat([train,trainT])
#    test=testT
#    re,model=f(train,test,na='stable_')
#    model.best_estimator_.alpha

    
    
    # 抽样拟合
    train,test  = train_test_split(finaldata, train_size=0.9, random_state=1)
    re,model=f(train,test,na='D_')
    #

    # model.predict(tmp.iloc[:,:-1])
    # modela = joblib.load("./model/stable__Model.pkl")
    # modela.predict(test)
    # model.best_estimator_.predict(test)
    pass





















