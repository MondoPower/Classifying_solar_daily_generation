import etl.influx_etl as influx_etl
from etl import util
import pandas as pd
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import tz
import itertools
import requests
import os, sys, glob, json
import s3
import pandas as pd
from datetime import datetime, timedelta
from dateutil import relativedelta
from etl import influx_etl, timeseries_etl, timestream_etl, error_handling
import pdb


def get_date_range(daterange):
    start_date = daterange["start_date"]
    if daterange["end_date"] is None:
        end_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        end_date = daterange["end_date"]
    return start_date, end_date
    
def get_location_full(location_id):
    daterange = {"start_date": '2022-08-19T00:00:00Z', "end_date": None}
    interval = 5
    influxETL = influx_etl.influx_etl()

    # get the location config,
    location_config, config = influxETL.get_config(
        location_id, return_raw=True)

    # get data for the above config from the influx database clone. This requires VPN connection
    # have to loop this over weeks to make it work
    start_date, end_date = get_date_range(daterange)
    #start_date = '2022-08-19T00:00:00Z'
    loc_data = influxETL.get_location_influx_data_historical(
        location_config, start_date, end_date, interval, verbose=True)
    loc_data['dates'] = pd.to_datetime(loc_data.dates)
    # convert to UTC as dates is in Melbourne time
    loc_data['time'] = pd.to_datetime(loc_data.dates, utc=True)

    return loc_data, location_config

def manipulate_weather_data(weather,df):

    weather.drop(columns=[
                        'solcastsiteid', 'postcode', 'periodtype', 'forecasttimeoffset'], inplace=True)
    pdb.set_trace()
    # combine with old stuff
    merged_df = pd.concat((weather, df))
    merged_df.drop_duplicates(inplace=True)

    # solcast nan issues, some lines are split, zero-fill, group, then sum to solve
    merged_df = merged_df.fillna(0.0)
    weather_df_cleaned = weather_df.groupby(['time']).sum()
    weather_df_cleaned.reset_index(inplace=True)
    weather_df_cleaned['time'] = pd.to_datetime(
        weather_df_cleaned.time, utc=True)
    return merged_data

def get_model_files():
    #get all the models from bucket
    bucket='prod-forecasting-infrastructurestac-model3d223d01-e9asr2kb6fi7'
    prefix='models/Demand_'
    s3_client= boto3.client('s3')
    objects = s3_client.list_objects_v2(Bucket=bucket,Prefix=prefix)
    for filename in objects['Contents']:
        if  filename['Key'].endswith('.tflite'):
            key=filename['Key']
            filename=key.split('/')[1]
            s3_client.download_file(Bucket=bucket,Key= key,Filename=f'Demand/{filename}')
        elif  filename['Key'].endswith('.tflite_metadata'): 
            key=filename['Key']
            filename=key.split('/')[1]
            s3_client.download_file(Bucket=bucket,Key= key,Filename=f'metaData/{filename}')           
    return 

  


