import etl.influx_etl as influx_etl
from etl import util
import pandas as pd
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import tz
import itertools
import requests
from bs4 import BeautifulSoup

def get_past_three_months():
    today = get_today()
    today_min1 = today - relativedelta(months=1)
    today_min2 = today - relativedelta(months=2)
    quarter_year = today_min1.year

    todays = [today, today_min1, today_min2]
    start_dates = []
    end_dates = []
    month_names = []
    for dt in todays:
        this_start_date, this_end_date, this_month_name = time(dt)
        start_dates.append(this_start_date)
        end_dates.append(this_end_date)
        month_names.append(this_month_name)
    return start_dates, end_dates, month_names, quarter_year

def get_today():
    today = datetime.now(tz=pytz.timezone('UTC'))
    return today

def time(today):
    # we want the month, and also enough information to build the start and end date datetime objects
    current_year = today.year
    current_month = today.month
    prev_month_datetime = today - relativedelta(months = 1)
    prev_year = prev_month_datetime.year
    prev_month = prev_month_datetime.month
    prev_month_name = prev_month_datetime.strftime("%B")

    # construct local times for the start and end date
    start_date_local = datetime(year = prev_year, month = prev_month, day = 1, hour = 0, tzinfo = tz.tzlocal())
    end_date_local = datetime(year = current_year, month = current_month, day = 1, hour = 23, tzinfo = tz.tzlocal()) -relativedelta(days=1)

    start_date_utc = start_date_local.astimezone(pytz.utc)
    end_date_utc = end_date_local.astimezone(pytz.utc)

    start_date_utc_str = start_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_utc_str = end_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    return start_date_utc_str, end_date_utc_str, prev_month_name

def scrape_bom_data():
    bom_url = "http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_nccObsCode=122&p_display_type=dailyDataFile&p_startYear=&p_c=&p_stn_num=086104"
    # import pdb; pdb.set_trace()
    header = headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like
    page = requests.get(bom_url, headers = header)
    soup = BeautifulSoup(page.content, "html.parser")

    table = soup.find_all('table')[0]
    column_names = table.thead.text.strip().split('\n')
    year = column_names[0]
    column_names = column_names[1:]
    data = {}
    for cc in column_names: data[cc] = []
    data['day'] = []

    for i, row in enumerate(table.tbody.find_all('tr')):
        if i > 0: # first column is irrelevant for us
            columns = row.find_all('td')
            for j, col in enumerate(column_names):
                data[col].append(columns[j].text.strip())
            data['day'].append(i)
    
    df = pd.DataFrame(data=data)
    df = df.iloc[0:-3] # get rid of the last threw rows as they are extra stats we don't want
    renamer = {"Jan": "January", 'Feb':'Febrary','Mar':'March','Apr':'April','Jun':'June','Jul':'July','Aug':'August','Sep':'September',
    'Oct':'October','Nov':'November','Dec':'December'}
    df.rename(columns = renamer, inplace=True)
    return df

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


  


