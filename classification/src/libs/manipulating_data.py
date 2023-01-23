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

def manipulate_weather_data(weather,df):

    weather.drop(columns=[
                        'solcastsiteid', 'postcode', 'periodtype', 'forecasttimeoffset','azimuth','zenith'], inplace=True)
    
    weather['time'] = pd.to_datetime(weather.time, utc=True)
    
    merged_df = df.merge(weather,on='time')
    merged_df.drop(columns=['pv_all'], inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df = merged_df.fillna(0.0)
    merged_df['date'] = merged_df.time.dt.strftime('%y-%m-%d')
    # average over day and then make day as index
   
    daily= merged_df.groupby('date')['air_temp'].mean()
    days_aggregated = pd.DataFrame(daily)
    days_aggregated.rename(columns={'air_temp':'average_tem'},inplace =True)
    df_new_day= merged_df.merge(days_aggregated, on='date')
    #df_new_daily = df_new.set_index('date')

    # max over a day

    daily_max = merged_df.groupby('date')['air_temp'].max()
    days_max = pd.DataFrame(daily_max)
    days_max.rename(columns={'air_temp':'max_tem'},inplace =True)
    df_max_day= df_new_day.merge(days_max, on='date')

     # min over a day

    daily_min = merged_df.groupby('date')['air_temp'].min()
    days_min = pd.DataFrame(daily_min)
    days_min.rename(columns={'air_temp':'min_tem'},inplace =True)
    df_min_day= df_max_day.merge(days_min, on='date')

    # midd of the day 

    idx = pd.date_range("2022-08-19", periods=24, freq="H")
    ts = pd.Series(range(len(idx)), index=idx)
    ts = pd.Series(range(len(idx)), index=idx)
    ts.resample("12H").mean()

    return df_min_day
 

def day_of_year(df_min_day):
    
    date_series = pd.Series(df_min_day['date'])
    date_series = date_series.map(lambda x: parse(x))
    day_year = date_series.dt.dayofyear
    day_year = pd.DataFrame(day_year)
    day_year.rename(columns = {'date':'d_y'}, inplace = True)
    df_min_day = pd.concat([day_year, df_min_day], axis = 1)
    
    return df_min_day

def month_of_year(df_min_day):
    pdb.set_trace()
    date_series = pd.Series(df_min_day['date'])
    date_series = date_series.map(lambda x: parse(x))
    month_year = date_series.dt.month
    month_year = pd.DataFrame(month_year)
    month_year.rename(columns = {'date':'m_y'}, inplace = True)
    df_min_day = pd.concat([month_year, df_min_day], axis = 1)

    return df_min_day


def doy(df_min_day):
 
    K=2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N

def year_month_day(Y,N):
   
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    M = int((9 * (K + N)) / 275.0 + 0.98)
    if N < 32:
        M = 1
    D = N - int((275 * M) / 9.0) + K * int((M + 9) / 12.0) + 30
    return Y, M, D

def statistical_labeling(df):
    
    pv_max = df.groupby(["time",'week_day'])["pv_all"].sum()
    print("Max pv %: ", df['pv_all'].max())
    print("Min pv %: ", df['pv_all'].min())
    # plt.title("Distribution in solar %")
    # df['pv_all'].hist()
    # plt.savefig('testplot.png')

    #labeling by hand
    df['pv_labeled'] = np.where(df['pv_all']>=4000, '1', '0')

    return df

def plot_models(merged_df):

    
    fig = go.Figure()
  
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df['pv_all'],mode='lines', name='label max', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scattergl(x=merged_df['time'], y=merged_df[merged_df['pv_labeled']==1], mode='lines', name='label_min', line=dict(color='blue',dash='dash')))
    fig.update_xaxes(title_text=f"{location} UTC")
    fig.update_yaxes(title_text=tcol)
    fig.show()

    return 

