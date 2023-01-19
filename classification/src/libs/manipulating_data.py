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

def return_high_low_solar(df):
    
    pv_max = df.groupby(["time",'week_day'])["pv_all"].sum()
    print("Max pv %: ", df['pv_all'].max())
    print("Min pv %: ", df['pv_all'].min())
    # plt.title("Distribution in solar %")
    # df['pv_all'].hist()
    # plt.savefig('testplot.png')

    #labeling by hand
    df['pv_labeled'] = np.where(df['pv_all']>=4000, '1', '0')
    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=df['week_day'], y=df[df['pv_labeled']==1],mode='lines', name='label max', line=dict(color='purple', dash='dash')))
    # fig.add_trace(go.Scattergl(x=df['week_day'], y=df[df['pv_labeled']==0], mode='lines', name='label_min', line=dict(color='orange')))

    # fig.show()
    return df
