import etl.influx_etl as influx_etl
from etl import util
import pandas as pd
import functools
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import tz
import itertools
from sklearn.preprocessing import StandardScaler
import requests
import os, sys, glob, json
import s3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import relativedelta
from etl import influx_etl, timeseries_etl, timestream_etl, error_handling
from tstools import manipulation

def prediction_reltime(row):
    return row.time + relativedelta.relativedelta(minutes = row.forecasttimeoffset)

def format_timestream_weather(df):
    df['prediction_time'] = df.apply(prediction_reltime, axis=1)
    manipulation.generate_cyclic_time_of_day(df, 'prediction_time', inplace = True)
    manipulation.generate_holidays(df, 'prediction_time', inplace = True)
    manipulation.generate_day_of_week(df, 'prediction_time', inplace = True)
    df.drop(columns=['time','forecasttimeoffset'], inplace=True)
    df.rename(columns = {"cloud_opacity": "cloudOpacity",
                            "week_day": "Day of week",
                            "holiday_flag": "Is_Holiday",
                            "sin_time": "Day sin",
                            "cos_time": "Day cos",
                            "prediction_time": "time"}, inplace = True)
    df["Is_Holiday"] = df['Is_Holiday'].astype(np.float64)
    df["Day of week"] = df["Day of week"].astype(np.float64)
    return df

def adjust_start_time(start_date):

    start_date = pd.to_datetime(start_date) - relativedelta.relativedelta(minutes = 15)
    end_date = start_date + relativedelta.relativedelta(minutes = 5)
    # put them in timestream format
    ts_start_date = start_date.strftime('%Y-%m-%d %H:%M:%S.%f')
    ts_end_date = end_date.strftime('%Y-%m-%d %H:%M:%S.%f')
    print(ts_start_date, ts_end_date)
    return ts_start_date, ts_end_date
    
def get_weather_data_from_timestream(start_date, horizon_hours = 24):
    
    horizon = (np.arange(12*horizon_hours, dtype=int)*5 + 10).tolist()
    ts_start_date, ts_end_date = adjust_start_time(start_date)
    df = timestream_etl.query_solcast_historical(ts_start_date, ts_end_date, None, verbose=True)
    match = (df.forecasttimeoffset >= horizon[0]) & (df.forecasttimeoffset <= horizon[-1])
    df = df[match]
    df['time'] = pd.to_datetime(df.time, utc=True)
    df.drop(columns = ['postcode', 'solcastsiteid', 'periodtype'], inplace = True)
    df_formatted = format_timestream_weather(df)
    df_formatted.drop(columns=['Day of week'],inplace=True)
    df_formatted.rename(columns={'Day sin':'Day_sin','Day cos':'Day_cos'},inplace=True)
    return df_formatted

def scaling_data(df_formatted):
    import pdb;pdb.set_trace()
    data = df_formatted.copy()
    data.drop(columns=['time'],inplace=True)
    
    scaler = StandardScaler()
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    data_normalized = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)
    

    print(data_normalized.describe().round(2))
    return data_normalized

def test_train_split(data_normalized, test_frac=0.15):
 
    data_normalized.reset_index(drop=True, inplace=True)
    split_loc = int(np.floor(len(data_normalized)*(1-test_frac)))
    x_train = data_normalized.loc[0:split_loc]
    x_test = data_normalized.loc[split_loc:]
    y_train = target.loc[0:split_loc]
    y_test = target.loc[split_loc:]
    
    return x_train, y_train,x_test,y_test

def manipulate_weather_data(final_df):
    
    final_df['time'] = pd.to_datetime(final_df.time, utc=True)
    final_df['date'] = final_df.time.dt.strftime('%y-%m-%d')
    final_df['hour'] = final_df.time.dt.hour
    final_df['month'] = final_df.time.dt.month
    final_df['doy'] = final_df.time.dt.dayofyear
    final_df['sin_doy'] = np.sin(2*np.pi*final_df.doy/365.0)
    final_df['cos_doy'] = np.cos(2*np.pi*final_df.doy/365.0)

    final_df.drop(columns=['Unnamed: 0','azimuth','zenith'], inplace=True)
    wcols_to_agg = ['air_temp', 'cloudOpacity', 'dhi', 'dni', 'dni10', 'dni90', 'ebh', 'ghi', 'ghi10', 'ghi90']

    # get just the points that happen 11-1?
    
    idx = np.where((final_df['hour']<=1) | (final_df['hour']>=11))
    weather_mid = final_df.loc[idx]
    
    # average weather features over day and then make day as index
    weather_day_ave = final_df.groupby('date')[wcols_to_agg + ['month', 'hour', 'doy', 'sin_doy', 'cos_doy']].mean()
    renamer = {col :col + '_mean' for col in wcols_to_agg}
    weather_day_ave.rename(columns = renamer, inplace=True)
    weather_day_ave.reset_index(inplace=True)
    
    # max over a day
    weather_day_max = final_df.groupby('date')[wcols_to_agg].max()
    renamer = {col :col + '_max' for col in wcols_to_agg}
    weather_day_max.rename(columns = renamer, inplace=True)
    weather_day_max.reset_index(inplace=True)

    # min over a day
    weather_day_min = final_df.groupby('date')[wcols_to_agg].min()
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
    
    final_weather = functools.reduce(lambda  left,right: pd.merge(left,right,on=['date'], how='outer'), weather_all)
    # maybe now merge with something else like the target label
    
    return final_weather



    