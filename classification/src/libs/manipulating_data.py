import os
import pandas as pd
import etl.influx_etl as influx_etl
import shutil
from etl import util
from datetime import datetime
from dateutil.relativedelta import relativedelta
import data_gathering
import urllib3
import requests
from dateutil import tz
import pytz
from time import strptime
import boto3
from tstools import manipulation
import pdb
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from dateutil.parser import parse
from functools import reduce
from sklearn.model_selection import GridSearchCV

def manipulate_data(df):
   
    df['time'] = pd.to_datetime(df.time)
    df['dates'] = pd.to_datetime(df.dates)

    PV_filter_control = [col for col in df.columns if "pv" in col and "_nc" not in col]
    df.dropna(subset=PV_filter_control, inplace=True)
    df.fillna(0, inplace=True)
    df["pv_all"] = df[PV_filter_control].sum(axis=1)
    df = manipulation.generate_day_of_week(df, 'time', inplace=False)
    df=manipulation.flag_daytime(df, utc_col = "time", city = 'Melbourne', inplace = True)
    cols_to_use = ['dates'] + ['pv_all'] + ['week_day'] + ['day_flag']
    df = df[cols_to_use]

    return df 

def manipulate_weather_data(weather):
    
    weather['time'] = pd.to_datetime(weather.time, utc=True)
    weather['date'] = weather.time.dt.strftime('%y-%m-%d')
    weather['hour'] = weather.time.dt.hour
    weather['month'] = weather.time.dt.month
    weather['doy'] = weather.time.dt.dayofyear
    weather['sin_doy'] = np.sin(2*np.pi*weather.doy/365.0)
    weather['cos_doy'] = np.cos(2*np.pi*weather.doy/365.0)

    weather.drop(columns=['solcastsiteid', 'postcode', 'periodtype', 'forecasttimeoffset','azimuth','zenith', 'time'], inplace=True)
    wcols_to_agg = ['air_temp', 'cloud_opacity', 'dhi', 'dni', 'dni10', 'dni90', 'ebh', 'ghi', 'ghi10', 'ghi90']

    # get just the points that happen 11-1?
    
    idx = np.where((weather['hour']<=1) | (weather['hour']>=11))
    weather_mid = weather.loc[idx]
    
    # average weather features over day and then make day as index
    weather_day_ave = weather.groupby('date')[wcols_to_agg + ['month', 'hour', 'doy', 'sin_doy', 'cos_doy']].mean()
    renamer = {col :col + '_mean' for col in wcols_to_agg}
    weather_day_ave.rename(columns = renamer, inplace=True)
    weather_day_ave.reset_index(inplace=True)
    
    # max over a day
    weather_day_max = weather.groupby('date')[wcols_to_agg].max()
    renamer = {col :col + '_max' for col in wcols_to_agg}
    weather_day_max.rename(columns = renamer, inplace=True)
    weather_day_max.reset_index(inplace=True)

    # min over a day
    weather_day_min = weather.groupby('date')[wcols_to_agg].min()
    renamer = {col :col + '_min' for col in wcols_to_agg}
    weather_day_min.rename(columns = renamer, inplace=True)
    weather_day_min.reset_index(inplace=True)
    # average weather features over day and then make day as index
    weather_mid_ave = weather_mid.groupby('date')[wcols_to_agg].mean()
    renamer = {col :col + '_mean_mid' for col in wcols_to_agg}
    weather_mid_ave.rename(columns = renamer, inplace=True)
    weather_mid_ave.reset_index(inplace=True)
    
    # max over a day
    weather_mid_max = weather_mid.groupby('date')[wcols_to_agg].max()
    renamer = {col :col + '_max_mid' for col in wcols_to_agg}
    weather_mid_max.rename(columns = renamer, inplace=True)
    weather_mid_max.reset_index(inplace=True)

    # min over a day
    weather_mid_min = weather_mid.groupby('date')[wcols_to_agg].min()
    renamer = {col :col + '_min_mid' for col in wcols_to_agg}
    weather_mid_min.rename(columns = renamer, inplace=True)
    weather_mid_min.reset_index(inplace=True)

    # merge everything on "date":
    weather_all = [weather_day_ave,weather_day_max,weather_day_min,weather_mid_ave,weather_mid_max,weather_mid_min]
    # final_weather = weather_day_ave.merge(weather_day_max).merge(weather_day_min).merge(weather_mid_ave).merge(weather_mid_max).merge(weather_mid_min)
    
    final_weather = reduce(lambda  left,right: pd.merge(left,right,on=['date'], how='outer'), weather_all)
    # maybe now merge with something else like the target label
    
    
    return final_weather

def get_hour(row):
    return row.hour

def get_date(row):
    return row.strftime('%y-%m-%d')

def manual_statistics(df):
   
    df['hour'] = df.dates.apply(get_hour)
    df['date'] = df.dates.apply(get_date)
   
    mask = df[(df['date'] >='22-06-01') & (df['date'] < '23-08-30')]
    df=mask.copy()
     
    # real weather data from bom website
    dec = [19.7,25.8,30.2,33.2,17.6,21.8,20.5,16.3,18.9,25.9,28.3,17.4,16.8,16.4,18.2,19.5,21.3,22.2,25.3,26,28.9,24.4,25.4,25.1,33,34.3,36.5,27.9,19.8,30.1,31.8]
    Jan = [35.6,32.6,21.2,18.3,24.3,25.0,29.6,33.0,29.9,26.3,34.6,28.5,29.6,37.1,21.5,32.0,36.1,18.6,19.7,24.4,27.5,27.7,29.2,26.4,31.9,24.2,30.4,34.0,24.0,23.0]
    feb = [21.8,21.2,17.6,20.9,20.3,22.7,23.4,24.6,30.4,31.5,30.3,21.8,21.6,24.4,31.8,36.2,39.9,25.6,29.3,29.2,23.2]
    june = [12.6,12.1,11.8,14.2,12.7,13,10.9,11.9,12.7,13,12,12.9,12.9,12.5,12.9,14.9,13.7,16.3,15.8,17.4,14.7,14.5,15.8,14.6,15.5,12.2,11.8,12.9,11.7,15.5]
    july = [13.2,13,12,13.4,14.3,16.4,13.2,11.9,12.2,16,13.9,9.7,13.3,11.8,14,13,14.8,11.9,12.6,13.1,17.1,18.1,18,15.9,11.2,14.9,13.3,12.8,13.1,15.1]
    august = [14,16,18.3,18.1,13.3,14.1,14.3,14.3,13.8,14.3,14.6,14.6,17.9,13.1,15.7,12.8,15,14.7,15.2,14.1,13.7,17,12,12.4,13.5,13.8,18.7,19.5,17.6,11.9,16.4]
    # labeling three months
    june = pd.DataFrame(june)
    june.rename(columns={0:'june'},inplace=True)
    july = pd.DataFrame(july)
    july.rename(columns={0:'july'},inplace=True)
    august = pd.DataFrame(august)
    august.rename(columns={0:'august'},inplace=True)
    winter = pd.concat([june,july,august],axis=1)
    winter.fillna(0,inplace=True)
    
    df.drop(columns=['week_day','day_flag','hour','dates'],inplace=True)
    df_date = df.groupby('date')['pv_all'].max()
    df_data=pd.DataFrame(df_date)
    df_data = df_data.reset_index()
    df2 = pd.melt(winter)
    df2.drop(columns=['variable'],inplace=True)
    df_mask = df_data[0:92]
    df_mask['winter_weather']= df2.value
   
    mean= df_mask['winter_weather'].mean()
    df_mask['pv_labeled'] = np.where(df_mask['winter_weather']>=mean , '1', '0')
    #df_data.reset_index()
    df = df_mask.copy()
    return df 


def statistical_labeling(df):
 
    df['hour'] = df.dates.apply(get_hour)
    df['date'] = df.dates.apply(get_date)

    idx = (df['hour']<=16) & (df['hour']>=11)
    df_mid = df[idx].copy()
    df_mid_ave = df_mid.groupby('date')['pv_all'].mean()
    df_mid_ave=pd.DataFrame(df_mid_ave)
   
    df_mid_ave.rename(columns = {'pv_all':'mean_mid_day'}, inplace=True)
    df_mid_ave.reset_index(inplace=True)
    
    pv_max = df["pv_all"].max()
    fraction = pv_max*0.3
    df_mid_ave['pv_labeled'] = np.where(df_mid_ave['mean_mid_day']>=fraction , '1', '0')
    df = df_mid_ave.copy()
    
    return df

def merged_data(manipulated_weather,df):
   
    #df_reset =df.reset_index()
    #df.drop(columns=['mean_mid_day'],inplace=True)
    #final_df['time'] = pd.to_datetime(manipulated_weather.time, utc=True)
 
    features_all = manipulated_weather.merge(df,on = ['date'])    
    features_all['labeled_target']=features_all.pv_labeled.astype(float)
    features_all.drop(columns=['pv_labeled', 'month', 'hour', 'doy','pv_all','winter_weather'],inplace=True)
   
    #correlation features
    # correlate = features_all.corr()
    new_sample = features_all[0:2]
    # correlate["labeled_target"].sort_values(ascending=False).iloc[0:10]   

    return features_all


    # maxi_mid_day['mean_mid_day'].plot(kind='barh')
    # plt.title("Distribution in solar %")
    # maxi_mid_day['mean_mid_day'].hist()
    # plt.savefig('new.png')
