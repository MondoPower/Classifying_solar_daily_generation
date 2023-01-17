import etl.influx_etl as influx_etl
from etl import util
import pandas as pd
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import tz
import itertools
import requests

import pandas as pd
from datetime import datetime, timedelta
from dateutil import relativedelta
from etl import influx_etl, timeseries_etl, timestream_etl, error_handling


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

def gather_dataset_for_location(location_id, include_month_lag = True, include_week_lag = True, include_day_lag = True, n_lags = 18, lag_weather_cols = True):
    file_exists = exists(f'raw_location_data/{location_id}_data.csv')
    if file_exists == False:
        loc_data, location_config = get_location_full(location_id)
        loc_data.to_csv(
            f'raw_location_data/{location_id}_data.csv', index=False)

    else:
        loc_data = pd.read_csv(f'raw_location_data/{location_id}_data.csv')
        start_date, end_date = get_date_range()
        loc_data['time'] = pd.to_datetime(loc_data.time)
        #loc_data = loc_data[(loc_data['time'] >= start_date) & (loc_data['time']<= end_date)]
    solcast = s3_helpers.read_csv_from_s3(
        'nonprod-automateddatawranglin-datastorage258d846d-1udpy7zvygt7d', 'automated-data-store/solcast_5min_horizon.csv')

    old_solcast_5min = s3_helpers.read_csv_from_s3(
        'mondo-edge-forecasting', 'experiments/random_forest/weather_sept_2021_to_march_2022.csv')
    weather_df = join_weather_datasets(old_solcast_5min, solcast)
    weather_cols = [cc for cc in weather_df.columns if 'time' not in cc]
    # join everything together then clean
    merged_df = join_dataframes(loc_data, weather_df, leave_pre_weather=False)
    aggregated_df = aggregate_source_type_cols(merged_df)
    # make lags: we want 3hours back so lag is 36, and the same time a day ago would be seasonsize = 288?
    gatecols = [cc for cc in aggregated_df.columns if 'gatemeter' in cc]

    cols_to_use = ['time'] + weather_cols + gatecols + ['pv']
    aggregated_df = aggregated_df[cols_to_use]
    
    if lag_weather_cols == True:
        columns_to_lag = ['pv', 'dhi', 'dni']
    else: columns_to_lag = ['pv']
    df_lagged = manipulation.generate_lags(
        n_lags, aggregated_df, columns_to_lag, seasonsize=1)
    columns_to_lag = ['pv']
    if include_day_lag == True:
        df_lagged = manipulation.generate_lags(
            1, df_lagged, columns_to_lag, seasonsize=288)
    if include_week_lag == True:    
        df_lagged = manipulation.generate_lags(
            1, df_lagged, columns_to_lag, seasonsize=2016)
    if include_month_lag == True: 
        df_lagged = manipulation.generate_lags(
            1, df_lagged, columns_to_lag, seasonsize=8064)

    cleaned_df = handle_missings(df_lagged)
    pv_training_cols = [cc for cc in cleaned_df if 'pv' in cc]
    weather_training_cols = [cc for cc in cleaned_df if 'cloud_opacity' in cc  or 'dhi' in cc or 'dni' in cc or 'dni90' in cc]
    weather_training_cols += weather_cols
    weather_cols = np.unique(weather_training_cols).tolist()

    final_df = cleaned_df.dropna(subset=pv_training_cols + weather_cols + ['time'])
    final_df = final_df[['time'] + pv_training_cols + weather_cols]

    # rename the pv columns to not have "_cleaned in them"
    return final_df, pv_training_cols, weather_cols


def identify_controllable_devices(location_id, config, start_date, end_date):
    raw_liveusage_data = timeseries_etl.get_device_data_from_liveusage(location_id, start_date, end_date)
    if len(raw_liveusage_data) == 0:
        live_usage_data = config[['deviceId', "nonControllable"]]
    else: live_usage_data = raw_liveusage_data.merge(config[['deviceId', "nonControllable"]].drop_duplicates(), on = 'deviceId', how = "left")
    return live_usage_data

def shape_loc_data_from_liveusage(location_id, config, start_date, end_date):
    """
    This functions should identify rows in the live usage response and 
    Extract them to make something like a dataframe that comes out of the mondo_ds_etl code
    
    Identify PV rows, battery power/SOC rows, gate rows, and handle controllability
    """
    live_usage_data = identify_controllable_devices(location_id, config, start_date, end_date)
    dfs_to_merge = []

    if 'metricType' in live_usage_data.columns:
        # this checks there's data at all, if there isn't we just skip everything and end up with an empty DataFrame
        data = live_usage_data[live_usage_data.metricType != 'ApparentPowerVA']
        n_pv = 1
    
        for _, row in data.iterrows():
            # we've found a PV inverter
            # add it to list of dataFrames and name the column appropriately
            col_name = None
            n_bats = 0
            n_gates = 0
            n_socs = 0

            this_data_stream = pd.DataFrame(row["dataPoints"])
            if row['nonControllable'] == False:
                c_flag = 'c'
            else: c_flag = 'nc'

            if (row['loadType'] == 'PvInverter') | (row['loadType'] == 'HybridPv'): 
                col_name = "pv_" + str(n_pv) + "_" + c_flag
                n_pv += 1

            if row['metricType'] == 'BatteryLevelPercentage'  and row['nonControllable'] == False:
                col_name = "soc" 
                n_socs += 1

            if (row['loadType'] == 'BatteryInverter') or (row['loadType'] == "HybridBattery" and row['metricType'] == "DirectCurrentPowerWatts"): 
                col_name = 'batt_' + c_flag
                n_bats += 1
            if row['loadType'] == 'GateMeter': 
                col_name = 'gatemeter'
                n_gates += 1
            
            if col_name == None:
                print("Something has gone wrong with column name identification")
                print("Human, please check")
                import pdb; pdb.set_trace()
            if (n_bats > 1) | (n_gates > 1) | (n_socs > 1):
                print("We have multiple batteries or gates or SOCs!")
                print("This might be a contract failure!")
                import pdb; pdb.set_trace()

            this_data_stream.rename(columns = {"value": col_name}, inplace = True)
            this_data_stream = this_data_stream.astype({col_name:'float'})
            dfs_to_merge.append(this_data_stream.copy())
    
    return merge_source_types(dfs_to_merge)

def merge_source_types(dfs_to_merge):
    merged_df = pd.DataFrame()
    for i in range(len(dfs_to_merge)):
        if i == 0:
            merged_df = dfs_to_merge[i]
        else:
            merged_df = merged_df.merge(dfs_to_merge[i])
    return merged_df

def aggregate_source_types(merged_df):
    pv_c = [col for col in merged_df.columns if "pv" in col and "_c" in col]
    if len(pv_c) > 0:
        merged_df['pv'] = merged_df[pv_c].sum(axis = 1)
    else: 
        raise error_handling.DataNotAvailable("cannot make prediction for pv model as no pv values present", merged_df.columns[merged_df.isna().any()].tolist())
    if merged_df.isna().values.any() == True:
        raise error_handling.DataNotAvailable("cannot make prediction as na values present", merged_df.columns[merged_df.isna().any()].tolist())
    merged_df.rename(columns = {"datetime":"timestamp"}, inplace = True)
    return merged_df

def get_location_data(location_id):
    start_date, end_date = get_date_range(18, 0)
    influxETL = influx_etl.influx_etl()
    location_config, config = influxETL.get_config(location_id, return_raw=True)
    merged_data = shape_loc_data_from_liveusage(location_id, location_config, start_date, end_date)
    datetime_data = aggregate_source_types(merged_data)
    datetime_data.rename(columns = {"timestamp":"time"}, inplace = True)
    return datetime_data

def main_data(start_date, end_date, interval = 60,influx_engine =None):
    """
    This function downloads all data for the specified time period
    """
    if influx_engine is None:
        influxETL = influx_etl.influx_etl()
    else: influxETL = influx_engine

    property_list = influxETL.get_hub_property_listings(community_name)
    property_list['property_id']=property_list.property_id.astype(int)
    property_list.sort_values('property_id',inplace=True,ignore_index=True)
    df_s = pd.DataFrame()
    for j in range(0,len(property_list['property_id'])):
        print(property_list['property_id'][j])
        meta_raw = influxETL.get_property_id_meta(property_list.iloc[j].property_id)
        location_config = util.parse_myubi_meta(meta_raw, property_list.iloc[j].property_id)
    
        loc_data = influxETL.get_location_influx_data_historical(location_config, start_date, end_date, interval, verbose=False, ignore_charge_lims=True, get_location_setpoint = False)
        ac_splits=[col for col in loc_data.columns if 'Split System' in col]
        main=[col for col in loc_data.columns if 'Mains Power' in col]
        cooktop=[col for col in loc_data.columns if 'Cooktop' in col]
        if (len(ac_splits)>1) | (len(main)>1) | (len(cooktop)>1):
            print('too many devices')
            import pdb;pdb.set_trace()

        if len(ac_splits)>0:
            loc_data.rename(columns={'location_id':'property_id', ac_splits[0]:'aircon',cooktop[0]:'hotplate',main[0]:'load'},inplace=True)

            df_s=pd.concat((df_s,loc_data[['dates','property_id','aircon','hotplate','load']]))
            df_s['aircon'] = df_s['aircon'].abs()
            df_s['hotplate'] = df_s['hotplate'].abs()
            df_s['load'] = df_s['load'].abs()

   
    return df_s, property_list


  


