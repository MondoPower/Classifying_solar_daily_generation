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
    pdb.set_trace()

    

    

   



    # use this colum method and then merge with original dataframe

    df_new= merged_df.merge(merged_df.groupby('date')['air_temp'].mean(), on='date')
    # daily = merged_df.resample('D').sum()
    # daily.reset_index(inplace=True)
    # daily['average_over_dayTem']= daily.groupby(['time'])['air_temp'].mean()
    # daily.drop_duplicates(inplace=True)
    # daily = daily.fillna(0.0)

    #resampling to monthly

    monthly = daily.resample('M').sum()


    # merged_df = merged_df.set_index('week_day')
  
    # merged_df_aggregate.reset_index(inplace=True)

    return merged_df

#  def add_day(merged_df):
    
#     dates=[]
#     for i in range(0,len(merged_df['time'])):
#         example_time=merged_df['time'].iloc[i]
#         mytimezone = example_time.astimezone(datetime.timezone(datetime.timedelta(days=10), name='GMT+10'))
#         dates.append(mytimezone)
#     output_df['time_NEM']=dates
    
#     return output_df

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

