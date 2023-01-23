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

def manipulate_data(df):

    df['time'] = pd.to_datetime(df.time)
    PV_filter_control = [col for col in df.columns if "pv" in col]
    df.dropna(subset=PV_filter_control, inplace=True)
    df.fillna(0, inplace=True)
    df["pv_all"] = df[PV_filter_control].sum(axis=1)
    df = manipulation.generate_day_of_week(df, 'time', inplace=False)
    df=manipulation.flag_daytime(df, utc_col = "time", city = 'Melbourne', inplace = True)
    cols_to_use = ['time'] + ['pv_all'] + ['week_day'] + ['day_flag']
    df = df[cols_to_use]

    return    df 

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
    
    # merged_df = df.merge(weather,on='time')
    # merged_df.drop(columns=['pv_all'], inplace=True)
    # merged_df.drop_duplicates(inplace=True)
    # merged_df = merged_df.fillna(0.0)
    # merged_df['date'] = merged_df.time.dt.strftime('%y-%m-%d')
    
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


# def day_of_year(df_min_day):
    
#     date_series = pd.Series(df_min_day['date'])
#     date_series = date_series.map(lambda x: parse(x))
#     day_year = date_series.dt.dayofyear
#     day_year = pd.DataFrame(day_year)
#     day_year.rename(columns = {'date':'d_y'}, inplace = True)
#     df_min_day = pd.concat([day_year, df_min_day], axis = 1)
#     df_min_day['sin_time'] = np.sin(df[time_column].map(datetime.datetime.timestamp)*2*np.pi/seconds_per_day)
#     df_min_day['cos_time'] = np.cos(df[time_column].map(datetime.datetime.timestamp)*2*np.pi/seconds_per_day)

#     return df_min_day




def statistical_labeling(df):
    
    
    df['date'] = df.time.dt.strftime('%y-%m-%d')
    df.drop(columns=['time','day_flag'], inplace=True)
    pv_max = df.groupby(["date"])["pv_all"].max()
    pv_min = df.groupby(["date"])["pv_all"].min()
    maximum_pv=pd.DataFrame(pv_max)
    maximum_pv.rename(columns={'pv_all':'maximum_daily'},inplace = True)
    maximum_pv.reset_index(inplace=True)
    minimum_pv=pd.DataFrame(pv_min)
    minimum_pv.rename(columns={'pv_all':'minimum_daily'},inplace = True)
    minimum_pv.reset_index(inplace=True)
    final_df = df.merge(minimum_pv).merge(maximum_pv)
    import pdb;pdb.set_trace()
    final_df['pv_labeled'] = np.where(final_df['pv_all']>=final_df['maximum_daily'], '1', '0')
    print("Max pv %: ", df['pv_all'].max())
    print("Min pv %: ", df['pv_all'].min())
    # plt.title("Distribution in solar %")
    # df['pv_all'].hist()
    # plt.savefig('testplot.png')

    #labeling by hand
    # this doesn't work quite as we want because it labels every 5 min chunk differently, even on the same day

    return df

def plot_models(merged_df):

    
    fig = go.Figure()
  
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df['pv_all'],mode='lines', name='label max', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df[merged_df['pv_labeled']==1], mode='lines', name='label_min', line=dict(color='blue',dash='dash')))
    fig.update_xaxes(title_text=f"{location} UTC")
    fig.update_yaxes(title_text=tcol)
    fig.show()

    return 

