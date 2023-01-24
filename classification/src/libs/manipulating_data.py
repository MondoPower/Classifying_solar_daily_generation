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

#
def manipulate_data(df):
    # import pdb; pdb.set_trace()
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
    
    # do the same for just the middle period of the day, but add more column name parts 
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

def statistical_labeling(df):
    
    df['hour'] = df.dates.apply(get_hour)
    df['date'] = df.dates.apply(get_date)
    # df.drop(columns=['dates','day_flag'], inplace=True)
    # average over peack hours
    #convert 
    idx = (df['hour']<=13) & (df['hour']>=11)
    df_mid = df[idx].copy()
    df_mid_ave = df_mid.groupby('date')['pv_all'].mean()
    df_mid_ave=pd.DataFrame(df_mid_ave)
    #if maximum daily greater than mid time average 
    df_mid_ave.rename(columns = {'pv_all':'mean_mid_day'}, inplace=True)
    df_mid_ave.reset_index(inplace=True)
    
    pv_max = df["pv_all"].max()
    fraction = pv_max*0.5
    df_mid_ave['pv_labeled'] = np.where(df_mid_ave['mean_mid_day']>=fraction , '1', '0')

    df = df.merge(df_mid_ave[['date','pv_labeled']], on = 'date')
    import pdb;pdb.set_trace()
    # maximum = final.groupby(["date"])["mean_mid_day"].max()

    # maxi_mid_day =pd.DataFrame(maximum)
    # #if maximum daily greater than mid time average 
    # maxi_mid_day.reset_index(inplace=True)
    
    # pv_min = df.groupby(["date"])["pv_all"].min()
    # maximum_pv=pd.DataFrame(pv_max)
    # maximum_pv.rename(columns={'pv_all':'maximum_daily'},inplace = True)
    # maximum_pv.reset_index(inplace=True)
    # minimum_pv=pd.DataFrame(pv_min)
    # minimum_pv.rename(columns={'pv_all':'minimum_daily'},inplace = True)
    # minimum_pv.reset_index(inplace=True)
    # mean_max = np.mean(maximum_pv['maximum_daily'], axis=0)
    # mean_min = np.mean(minimum_pv['minimum_daily'], axis=0)

    # final_df = minimum_pv.merge(maximum_pv).merge(df_max_mid)
    
    # maxi_mid_day['mean_mid_day'].plot(kind='barh')
    # plt.title("Distribution in solar %")
    # maxi_mid_day['mean_mid_day'].hist()
    # plt.savefig('new.png')

    #labeling by hand
    # this doesn't work quite as we want because it labels every 5 min chunk differently, even on the same day

    return final

def merged_data(final_weather,final):
  
    final.drop(columns=['mean_mid_day','pv_all','week_day','hour'],inplace=True)

    features_all = final_weather.merge(final)

    dfCorr = features_all.corr().abs()
    to_keep = [c for c in dfCorr.columns if dfCorr[c] > 0.9]
    
    plt.figure(figsize=(30,10))
    sn.heatmap(filteredDf, annot=True, cmap="Reds")
    plt.show()
    print(features_all.corr())

    return merged_df


def plot_models(merged_df):

    
    fig = go.Figure()
  
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df['pv_all'],mode='lines', name='label max', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df[merged_df['pv_labeled']==1], mode='lines', name='label_min', line=dict(color='blue',dash='dash')))
    fig.update_xaxes(title_text=f"{location} UTC")
    fig.update_yaxes(title_text=tcol)
    fig.show()

    return 

def make_eyeball_trace(df):


    fig = go.Figure()
  
    fig.add_trace(go.Scattergl(x=df['dates'], y=df['pv_all'],mode='lines', name='PV', line=dict(color='red')))
    fig.add_trace(go.Scattergl(x=df['dates'], y=df['pv_labeled'].astype(float).values * 5000, mode='lines', name='Label', line=dict(color='blue',dash='dash')))
    fig.show()