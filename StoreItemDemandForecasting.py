# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:43:36 2021

@author: kohxz

Dataset: kaggle competitions download -c demand-forecasting-kernels-only
Description: Predict 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores.

Note: ARIMA, SARIMA(365) took too long to train. we will focus on ML forecasting.
"""
import os
os.chdir("C:\\Users\\kohxz\\Desktop\\Sandbox\\Projects\\Fairprice_PredictSales\\")
print(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import datetime 
datetime.datetime.strptime
import calplot

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
import scipy.stats as scs

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

sns.set_theme(style="darkgrid")

#readfile
train0 = pd.read_csv('demand-forecasting-kernels-only\\train.csv', parse_dates= ['date'])
valid0 = pd.read_csv('demand-forecasting-kernels-only\\test.csv', parse_dates= ['date'])

#check data shape 
train0.shape
valid0.shape 

#check data type
train0.info()
valid0.info()

# #change data type
# for df in [train0, valid0]:
#     for col in ['item', 'store']:
#         df[col] = df[col].astype('str')
        
#check missing values !no missing value 
print(train0.isnull().sum())
print(valid0.isnull().sum())

#check unique value, stats
for col in train0.columns:
    print('# of unique values in {}: {}'.format(col, train0[col].nunique()))

train0.describe()

#check dates
print('Train Min Date: {}'.format(min(train0['date']))) #2013-01-01
print('Train Max Date: {}'.format(max(train0['date']))) #2017-12-31
print('Test Min  Date: {}'.format(min(valid0['date']))) #2018-01-01
print('Test Max  Date: {}'.format(max(valid0['date']))) #2018-03-31

#combine train and test dataset
data0 = pd.concat([train0, valid0], axis=0, sort=False)
data0 = data0.reset_index()
# =============================================================================
# Feature Engineering
# =============================================================================
print("# of items {}".format(data0['item'].nunique())) #!0 to 49 items
print("# of store {}".format(data0['store'].nunique()))#!1 to 10 stores 
    
#add new columns
data1 = data0.copy()
data1['year'] = data1['date'].dt.year
data1['quarter'] = data1['date'].dt.quarter
data1['month'] = data1['date'].dt.month
data1['weekday'] = data1['date'].dt.dayofweek+1 #monday=1, sunday=7
data1['is_weekend'] = np.where(data1['weekday']>=6, 1, 0)
data1['is_monthend'] = np.where(data1['date'].dt.is_month_end==True, 1,0)
data1['is_monthstart'] = np.where(data1['date'].dt.is_month_start==True, 1,0)
data1['weekyear'] = data1['date'].dt.isocalendar().week 

#remove leap year Feb29 
data1['is_leapyear'] = (np.where(data1['date'].dt.is_leap_year==True,1,0)).astype('object')
data1['dayyear'] = data1['date'].dt.dayofyear
data1= data1.loc[~((data1['is_leapyear']==1) & (data1['dayyear']==60))]
data1.drop(columns=['dayyear', 'is_leapyear'], inplace=True)

data1['year_quarter'] = data1['year']*100 + data1['quarter'] #for visualize
data1['year_month'] =  data1['year']*100 + data1['month'] #for visualize

data1.info()
# =============================================================================
# EDA - Visualization
# =============================================================================
#create temp table for visualization excluding validation database in 2018
data1a = data1[~data1['sales'].isna()]

# Distribution plot of sales 
plt.title('Distribution of sales by item, store and date')
sns.distplot(data1a['sales'])

plt.title('Distribution of sales (log) by item, store and date')
sns.distplot(np.where(data1a['sales']==0, 0, np.log1p(data1a['sales'])))

# Distribution plot of sales by item
plt.title('Distribution of sales by store')
sns.boxplot(x='store', y='sales', data=data1a)
plt.xticks(rotation=90)

plt.title('Distribution of sales by item')
sns.boxplot(x='item', y='sales', data=data1a)
plt.xticks(rotation=90)

#check if there is any missing dates
data1a.loc[:, ['date', 'store']].groupby('store').count().reset_index()
data1a.loc[:, ['date', 'item']].groupby('item').count().reset_index()

#---------------------------------------------------------------------------
def PlotBar(df, xcol, ycol, title=None) :
    df_temp = df.loc[:, ['sales'] + [xcol]]\
                        .groupby([xcol])\
                        .mean()\
                        .reset_index() #.sort_values(by=['sales'], ascending=False)\
   
    if title != '':
        plt.title('Average Sales by {}'.format(xcol))
    else:
        plt.title('{}'.format(title))                                   
    
    if xcol == 'month':
        ax= sns.barplot(data=df_temp, x=xcol, y='sales', palette="Blues_d", order=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    else:
        ax= sns.barplot(data=df_temp, x=xcol, y='sales', palette="Blues_d")
    ax.set_ylabel(ycol)
    ax.set_xlabel(xcol)
    if df_temp.shape[0] > 12:
        plt.xticks(rotation=90)
    plt.show()

def PlotCalendar(df, year):
    df_cal = df.loc[df['year']==year, ['sales', 'date']]\
                    .groupby(['date']).mean().reset_index()
    df_cal.index = df_cal['date']
    df_cal.drop(columns=['date'], inplace=True)
    
    calplot.calplot(df_cal['sales'])
    
#plot total sales by store & item (bar chart)
PlotBar(data1a, 'store', 'sales')

#plot top 10 and bottom 10 items 
df_item  = data1a.loc[:, ['item', 'sales']].groupby('item').mean().nlargest(10, 'sales').reset_index()
PlotBar(df_item, 'item', 'sales', 'Top 10 Sales by item')
df_item  = data1a.loc[:, ['item', 'sales']].groupby('item').mean().nsmallest(10, 'sales').reset_index()
PlotBar(df_item, 'item', 'sales', 'Botton 10 Sales by item')

#plot average sales by year, quarter, month, day of week, weekend
select_cols= ['year', 'quarter', 'month', 'weekday', 'is_weekend']

for viewcol in select_cols:
    print(">>> Plotting average sales by {}".format(select_cols))
    #plot avergae sales (bar chart)
    PlotBar(data1a, viewcol, 'sales')

#plot yearly sales heatmap by month, weekday
for yr in np.arange(2013, 2018).tolist():
    PlotCalendar(data1a, yr)

#---------------------------------------------------------------------------
def PlotLine (df, xcol, ycol, hue=None):
    plt.figure(1 , figsize = (15 , 10))
    plt.title('Distribution of {} by {}'.format(ycol, hue))
    ax = sns.lineplot(data=df, x=xcol, y=ycol, hue=hue, markers="o") 
    ax.set_ylabel(ycol)
    ax.set_xlabel(xcol)
    plt.xticks(rotation=90)
    plt.show()
    
select_col = ['year', 'year_quarter','year_month']

#convert selected fields to str datatype
for col in select_col:
    data1a[col] = data1a[col].astype('str')
data1a.info()

for viewcol in ['store', 'item']:
    for aggcol in select_col:
        print(">>> Aggregation at {} by {}".format(aggcol, viewcol))
        
        df_temp = data1a.loc[:, [viewcol] + ['sales'] + [aggcol]]\
                        .groupby([viewcol] + [aggcol])\
                        .agg({'sales':['mean', 'median', 'sum', 'count']})\
                        .sort_values(by=aggcol, ascending=True)\
                        .reset_index()
                        
        df_temp.columns = df_temp.columns.droplevel(level=1)
        df_temp.columns = [viewcol] + [aggcol] + ['average_sales', 'median_sales', 'total_sales', 'count']
        
        for selectmetric in ['average_sales']: #df_store.iloc[:, -4:-1].columns:
            #Plot trend 
            PlotLine(df_temp, aggcol, selectmetric, viewcol) 

#plot total sales and YOY growth %
for viewcol in ['store', 'item']:
    df_sales = data1a.loc[:, ['year','sales']+[viewcol]].groupby(['year']+[viewcol]).sum().reset_index()
    df_sales = pd.pivot_table(df_sales, values='sales', index=[viewcol], columns='year', fill_value=0)
    df_sales2= df_sales.copy()
    
    for yr in df_sales.iloc[:, -4:].columns:
        print(yr)
        df_sales2[yr]= ((df_sales2[yr]/df_sales[str(int(yr)-1)])-1)*100
    df_sales2.drop(columns=['2013'], inplace=True)
    
    df_sales2 = df_sales2.T.reset_index()
    df_sales2 = pd.melt(df_sales2, id_vars='year') 
    df_sales2.rename(columns={'value':'change%'}, inplace=True)
    
    plt.title('YOY Sales % Growth by {}'.format(viewcol))
    ax = sns.lineplot(data=df_sales2, x='year', y='change%', hue=viewcol, markers="o", legend=False) 
    plt.show()

del df_sales, df_sales2

# =============================================================================
# Build traditional time series - ARIMA model 
# create one model per store, item and test using last 3M of train dataset
# !!!! currently ARIMA does not work !!!! it is for data exploration only
# =============================================================================
ts_sig_all = pd.DataFrame()

for s in np.arange(1,2): #store #change it to 1, 11
    for i in np.arange(1,4): #item #change it to 1, 51
        print(">> store 1 and item {}".format(i))
        #test using 1 store with 50 items
        
        data1b = data1a.loc[(data1a['store']== s) & (data1a['item']==i) & 
                            (data1a['date'].dt.date < datetime.date(2017,10,1)) 
                            , ['date', 'item','store', 'sales']]
        data1b = data1b.loc[data1b['date'].dt.date != datetime.date(2016,2,29)]
        #check no leap year
        data1b['year'] = data1b['date'].dt.year
        data1b.loc[:, ['year', 'date']].groupby('year').count()
        data1b.drop(columns='year', inplace=True)
        
        test2 = data1a.loc[(data1a['store']==s) & (data1a['item']==i) & 
                             (data1a['date'].dt.date >= datetime.date(2017,10,1)), ['date', 'item','store', 'sales']]
        
        # valid2 = valid0.loc[(valid0['store']=='store' + str(s)) & (valid0['item']=='item' + str(i)) 
        #                    , ['date', 'item','store']]
        # valid2.index=valid2['date']
        
        #plot time series 
        ts = data1b.groupby(["date"])["sales"].sum()
        ts_test = test2.groupby(["date"])["sales"].sum()
        plt.figure(figsize=(16,8))
        plt.title('Total Sales')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.plot(ts)
        plt.plot(ts_test)
        
        #plot the predictions for validation set
        plt.plot(ts, label='Train')
        plt.plot(ts_test, label='Valid')
        plt.show()
        
        
        fig,ax = plt.subplots(2,1,figsize=(20,10))
        fig = sm.graphics.tsa.plot_acf(ts.diff().dropna(), lags=40, ax=ax[0])
        plt.title('ACF for store {} and item {}'.format(s, i))
        fig = sm.graphics.tsa.plot_pacf(ts.diff().dropna(), lags=40, ax=ax[1])
        plt.title('PACF for store {} and item {}'.format(s, i))
        plt.show()
    
  
        #ACF, PACF values 
        ts_sig = pd.DataFrame()
        ts_sig['model'] = "item" + str(i)
        ts_sig['acf (MA)'] = acf(ts.diff().dropna(), nlags=40)
        ts_sig['pacf (AR)'] = pacf(ts.diff().dropna(), nlags=40, method='ols')
        ts_sig['acf_sig'] = np.where(abs(ts_sig['acf (MA)'])>0.05, 1,0)
        ts_sig['pacf_sig'] = np.where(abs(ts_sig['pacf (AR)'])>0.05, 1,0)
        
        ts_sig
        ts_sig_all = pd.concat([ts_sig, ts_sig_all])
        
        # #build Arima model !-took too long to run
        # resDiff = sm.tsa.arma_order_select_ic(ts, max_ar=7, max_ma=7, ic='aic', trend='c')
        # print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')
        
        # arima = sm.tsa.statespace.SARIMAX(ts,order=(7,1,7),freq='D',seasonal_order=(0,0,0,0),
        #                          enforce_stationarity=False, enforce_invertibility=False,).fit()
        # arima.summary()

del ts_sig, ts_sig_all, ts_test, ts, i, s
del data1a, data1b

#--------------------------------------------------------------------
#Feature engineering on Sales 

def LagSales(df, lag):
    for i in lag:
        df['sales_lagdays' + str(i)] = df.loc[:, ['store', 'item', 'sales']]\
                                        .groupby(['store','item'])['sales']\
                                        .transform(lambda x: x.shift(i))
    return df

def MovingAverageSales(df, window):
    for i in window:
        df['sales_movingavgdays' + str(i)] = df.loc[:, ['store', 'item', 'sales']]\
                                        .groupby(['store','item'])['sales']\
                                        .transform(lambda x: x.shift(90).rolling(window=i).mean())
    return df

def MeanEncodingSales(df):
    for i in ['item', 'store']:
        df_agg_sales = pd.DataFrame()
        key = df[i].drop_duplicates().tolist()

        for aggfield in key:
            print(aggfield)
            #aggregate at year
            df_agg_sales_temp = df.loc[df[i]==aggfield, [i, 'year', 'sales']]\
                                    .groupby([i, 'year'])['sales']\
                                    .mean().reset_index()
            df_agg_sales_temp['sales_' + str(i) + '_prevyear'] = df_agg_sales_temp['sales'].shift()
            
            df_agg_sales=df_agg_sales.append(df_agg_sales_temp)
            
        df_agg_sales.drop(columns=['sales'], inplace=True)
        df_agg_sales.dropna(inplace=True)

        df = df.merge(df_agg_sales, how='left', on=[i, 'year'])
    return df

def MeanEncodingSales_wkday(df):
    for i in ['item', 'store']:
        df_agg_saleswd = pd.DataFrame()
        key = df[i].drop_duplicates().tolist()

        for aggfield in key:
            print(aggfield)
           
            #aggregate at year, month, weekday
            df_agg_saleswd_temp = df.loc[df[i]==aggfield, [i, 'year', 'month', 'weekday', 'sales']]\
                                    .groupby([i, 'year', 'month', 'weekday'])['sales']\
                                    .mean().reset_index()
            df_agg_saleswd_temp['year' +str(0)] = df_agg_saleswd_temp['year']          
            df_agg_saleswd_temp['year'] = (df_agg_saleswd_temp['year']).astype('int')+1     
            
            df_agg_saleswd=df_agg_saleswd.append(df_agg_saleswd_temp)

        df_agg_saleswd.drop(columns=['year0'], inplace=True) #for checking
        df_agg_saleswd.rename(columns={'sales':'sales_'+str(i)+'_prevyearwd'}, inplace=True)
        df_agg_saleswd.dropna(inplace=True)
        df_agg_saleswd['year'] = df_agg_saleswd['year']
        
        df = df.merge(df_agg_saleswd, how='left', on=[i, 'year', 'month', 'weekday'])
    return df
 
#add lag sales - min 90 lags for test dataset
data2 = LagSales(data1, [90, 91, 92, 93,  97, 120, 180, 365])

#add moving average sales 
data2 = MovingAverageSales(data1, [90, 120, 180, 210, 270, 365])

#add average sales of previous year @ item and store level
data2 = MeanEncodingSales(data1)

#add average sales of previous year, month, weekday @ item and store level
data2 = MeanEncodingSales_wkday(data1)

data2.info()
#data2.to_csv('data2.csv')
# =============================================================================
# Build ML Model
# =============================================================================
data2.columns 
data3 = pd.get_dummies(data2, columns=['store', 'item', 'quarter', 'weekday', 'month'])
data3['weekyear']=data3['weekyear'].astype('int')
# data3['is_weekend']=data3['is_weekend'].astype('int')
# data3['is_monthend']=data3['is_monthend'].astype('int')
# data3['is_monthstart']=data3['is_monthstart'].astype('int')

train = data3[(data3['date'].dt.date >= datetime.date(2014,1,1)) & (data3['date'].dt.date <= datetime.date(2017,10,31))]
test =  data3[(data3['date'].dt.date >= datetime.date(2017,10,1)) & (data3['date'].dt.date <= datetime.date(2017,12,31))]
valid = data3[data3['date'].dt.date >= datetime.date(2018,1,1)]
valid.isna().sum()

train.columns

x_col = [col for col in train.columns if col not in ['index', 'date', 'sales', 'id', 'year','year_quarter', 'year_month']]
y_col = ['sales']

X_train = train[x_col]
Y_train = train[y_col]
X_test = test[x_col]
Y_test = test[y_col]
X_valid = valid[x_col]

#build lightGBM
lgb_params = { 'objective': 'mse',
              'metric': 'rmse',
              'num_leaves': 100,
              'learning_rate': 0.005,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              #'max_depth': 5,
              'verbose': 1,
              'seed': 11,
              'num_boost_round': 15000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=x_col)
lgbval = lgb.Dataset(data=X_test, label=Y_test, reference=lgbtrain, feature_name=x_col)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  verbose_eval=200)

# get test errors
from sklearn.metrics import mean_squared_error 

y_pred_test = model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred_test)
print(">>> mean_squared_error in TEST dataset: {}".format(mse))
rmse = mean_squared_error(Y_test, y_pred_test, squared = False)
print(">>> root_mean_squared_error in TEST dataset: {}".format(rmse))

#plot feature importances
lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

lgb_params = {'objective': 'mse',
              'metric': 'rmse',
              'num_leaves': 100,
              'learning_rate': 0.01,
              'feature_fraction': 0.8,
              #'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=x_col)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
valid_preds = final_model.predict(X_valid, num_iteration=model.best_iteration)


#submit result 
df_submission = pd.DataFrame({
    "id": valid['id'], 
    "sales": valid_preds
})
df_submission.to_csv('submission.csv', index=False)

