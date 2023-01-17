from calendar import month_name
import os
from bs4 import BeautifulSoup
import pandas as pd
import etl.influx_etl as influx_etl
import shutil
from etl import util
from datetime import datetime
from dateutil.relativedelta import relativedelta
from libs import data_gathering
import urllib3
import requests
from dateutil import tz
import pytz
from time import strptime
import boto3
import tempfile



def generate_monthly_statistics(df_s, property_df):
    """
    This function takes location timeseries and produces averages of different powers, then writes to Excel file
    """
    # main load summary
    df_s['Date'] = df_s['dates']
    df_s['Date'] = df_s['Date'].dt.date
    month_summary = df_s.groupby(['property_id','Date'])['load'].sum()
    month_summary = month_summary.to_frame()
    month_summary.reset_index(inplace=True)
    month_summary['load'] = month_summary['load']/1000
    month_summary = pd.merge(month_summary,property_df[['property_id','lot_no']],on='property_id', how='left')
    month_summary = month_summary[['property_id', 'lot_no', 'Date', 'load']]
    month_summary

    #Average Daily usage per customer
    avg_cust = month_summary.groupby(['property_id'])['load'].mean().reset_index()
    avg_cust.columns = ['property_id', 'average_load']
    avg_cust = pd.merge(avg_cust,property_df[['property_id','lot_no']],on='property_id', how='left')
    avg_cust = avg_cust[['property_id', 'lot_no', 'average_load']]

    #Total monthly usage per customer
    total_cust = month_summary.groupby(['property_id'])['load'].sum().reset_index()
    total_cust.columns = ['property_id', 'total_load']
    total_cust = pd.merge(total_cust,property_df[['property_id','lot_no']],on='property_id', how='left')
    total_cust = total_cust[['property_id','lot_no', 'total_load' ]]
    total_cust.sort_values('total_load', inplace=True, ignore_index=True)


    #Total monthly usage as AVIVA by day of month
    daily_total = month_summary.groupby(['Date'])['load'].sum().reset_index()
    daily_total.columns = ['Date', 'daily_total_load']

    #Aircon consumption per day in the month per cust

    month_summary_aircon = df_s.groupby(['property_id','Date'])['aircon'].sum()
    month_summary_aircon = month_summary_aircon.to_frame()
    month_summary_aircon.reset_index(inplace=True)
    month_summary_aircon['aircon'] = month_summary_aircon['aircon']/1000
    month_summary_aircon = pd.merge(month_summary_aircon,property_df[['property_id','lot_no']],on='property_id', how='left')
    month_summary_aircon = month_summary_aircon[['property_id','lot_no', 'Date','aircon']]
    month_summary_aircon

    #average daily aircon per customer
    avg_cust_aircon = month_summary_aircon.groupby(['property_id'])['aircon'].mean().reset_index()
    avg_cust_aircon.columns = ['property_id', 'average_aircon']

    #Total monthly usage as AVIVA per day
    total_cust_aircon = month_summary_aircon.groupby(['Date'])['aircon'].sum().reset_index()
    total_cust_aircon.columns = ['Date', 'daily_total_aircon']

    #Oven/Hotplate consumption per day in the month per customer
    month_summary_hotplate = df_s.groupby(['property_id','Date'])['hotplate'].sum()
    month_summary_hotplate = month_summary_hotplate.to_frame()
    month_summary_hotplate.reset_index(inplace=True)
    month_summary_hotplate['hotplate'] = month_summary_hotplate['hotplate']/1000
    month_summary_hotplate = pd.merge(month_summary_hotplate,property_df[['property_id','lot_no']],on='property_id', how='left')
    month_summary_hotplate = month_summary_hotplate[['property_id','lot_no', 'Date','hotplate']]

    #average daily hotplate per customer
    avg_cust_hotplate = month_summary_hotplate.groupby(['property_id'])['hotplate'].mean().reset_index()
    avg_cust_hotplate.columns = ['property_id', 'average_hotplate']

    #Total monthly hotplate usage as AVIVA
    # total_cust_hotplate = month_summary_hotplate.groupby(['Date'])['hotplate'].sum().reset_index()
    # total_cust_hotplate.columns = ['Date', 'daily_total_hotplate']
    # total_cust_hotplace.sort_values('daily_total_hotplate', inplace=True, ignore_index=True)

    #Average daily AVIVA combined df for Main,Aircon, Hotplate by day of month - page 4
    avg_daily = avg_cust.copy()
    avg_daily['average_aircon'] = avg_cust_aircon['average_aircon']
    avg_daily['average_hotplate'] = avg_cust_hotplate['average_hotplate']
    avg_daily = avg_daily[['property_id', 'lot_no', 'average_load', 'average_aircon', 'average_hotplate']]
    avg_daily.sort_values('average_load', inplace=True, ignore_index=True)

    return avg_daily, total_cust, daily_total

def write_monthly_report(community_name):
    #influxETL = influx_etl.influx_etl(Database='legacy_meta')
    today=data_gathering.get_today()
    start_date,end_date,month_name=data_gathering.time(today)
    df_s,property_df=data_gathering.main_data(community_name, start_date, end_date,influx_engine = None)
    avg_daily, total_cust, daily_total = generate_monthly_statistics(df_s, property_df)

    template_file = os.path.join('reports', 'template_report.xlsx')
    temp_dir = tempfile.TemporaryDirectory()
    report_name = os.path.join(temp_dir.name, community_name + '_monthly_report_' + month_name + '_formatted.xlsx')
    report_save_name = community_name + '_monthly_report_' + month_name + '_formatted.xlsx'
 
    shutil.copyfile(template_file, report_name)

    # Define the writer in overlay mode
    
    writer1 = pd.ExcelWriter(report_name, engine='openpyxl', mode='a', if_sheet_exists = 'overlay')

    # write the three chunks of summary data to the correct cells
    avg_daily[['lot_no', 'average_load', 'average_aircon', 'average_hotplate']].to_excel(writer1, sheet_name = "Summary", index=False, header=False, startrow = 2, startcol = 5)
    total_cust[['lot_no','total_load']].to_excel(writer1, sheet_name = "Summary", index=False, header=False, startrow = 2, startcol = 11)
    daily_total[['Date','daily_total_load']].to_excel(writer1, sheet_name = "Summary", index=False, header=False, startrow = 2, startcol = 15)
    
    df_name = pd.DataFrame(data = {"community_name": [community_name]})
    year_num = (datetime.now() - relativedelta(months=1)).year
    df_month = pd.DataFrame(data = {"month_name": [month_name + " " + str(year_num)]})
    
    df_month[["month_name"]].to_excel(writer1, sheet_name = "Summary", index=False, header=False, startrow = 4, startcol = 3)
    df_name[["community_name"]].to_excel(writer1, sheet_name = "Summary", index=False, header=False, startrow = 0, startcol = 0)
    
    writer1.close()  
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(report_name,'aviva-monthly-reports', report_save_name)  

    return 

def make_date(row, month_name, year):
        try:
            date = datetime(year = year, month = strptime(month_name,'%B').tm_mon, day = row.day)
        except ValueError:
            date = datetime(year = 2200, month = 1, day = 1)
        return date

def quarterly_report(community_name):

    today=data_gathering.get_today()
    month_name=data_gathering.time(today)
    start_dates, end_dates, month_names, this_year= data_gathering.get_past_three_months()

    avg_daily_dfs = []
    total_cust_dfs = []
    daily_total_dfs = []
    daily_lot_avg_dfs = []
    daily_total_qtr=pd.DataFrame()
    influxETL = influx_etl.influx_etl(Database='legacy_meta')
    for i in range(len(start_dates)):
        start_date = start_dates[i]
        end_date = end_dates[i]
        

        df_s,property_df=data_gathering.main_data(community_name, start_date, end_date, influx_engine = influxETL)
        avg_daily, total_cust, daily_total = generate_monthly_statistics(df_s, property_df)
        avg_daily_dfs.append(avg_daily.copy()) 
        total_cust_dfs.append(total_cust.copy())
        daily_total_dfs.append(daily_total.copy())
       
    daily_total_qtr = pd.concat(daily_total_dfs)
    daily_total_qtr.sort_values('Date',inplace=True)

    max_temp=data_gathering.scrape_bom_data()
    
    whole_tem= pd.DataFrame()
    for i, month in enumerate(month_names):
        cols_weather=[cc for cc in max_temp.columns if month in cc]
        max_temp_month=max_temp[cols_weather+['day']].copy()

        max_temp_month['Date']= max_temp_month.apply(make_date, axis=1, args = (month, this_year))
        max_temp_month.rename(columns = {month: 'Max_Temperature'}, inplace=True)
        max_temp_month.drop(columns = ['day'], inplace = True)
        whole_tem = pd.concat((whole_tem, max_temp_month))

    daily_total_qtr['Date'] = pd.to_datetime(daily_total_qtr.Date)
    whole_tem['Date'] = pd.to_datetime(whole_tem.Date)
    final_max_tem=whole_tem.merge(daily_total_qtr, on='Date',how='left')
    final_max_tem.sort_values('Date',inplace=True)
    final_max_tem.dropna()
    final_max_tem = final_max_tem[final_max_tem.Date < datetime(year = 2100, month=1, day=1)]
    final_max_tem['Date'] = final_max_tem['Date'].dt.strftime('%Y-%m-%d')
    final_max_tem['Max_Temperature']=pd.to_numeric(final_max_tem['Max_Temperature'])
    final_max_tem.fillna(method='ffill',inplace=True)
    
    full_avg_by_lot = pd.concat(avg_daily_dfs).groupby(['lot_no', 'property_id'])['average_load'].mean().reset_index()
    
    for n, month in enumerate(month_names): 
        avg_daily_load=avg_daily_dfs[n].rename(columns = {'average_load': month}, inplace = False)
        full_avg_by_lot = full_avg_by_lot.merge(avg_daily_load[['lot_no', month]], on = 'lot_no', how = 'left')
    
    # making month names for the average use 
    df_month = pd.DataFrame()
    months=[]
    
    for month in month_names:
        months.append(month)
    df_months=pd.DataFrame(months)
    df_months.rename(columns={0:'month_name'},inplace=True)
    
    template_file_quarterly = os.path.join('reports', 'template_report_quarterly.xlsx')
    temp_dir = tempfile.TemporaryDirectory()  
    report_name_quarter = os.path.join(temp_dir.name, community_name + '_quarterly_report_formatted.xlsx')
    report_save_name = community_name + '_quarterly_report_formatted.xlsx'
    shutil.copyfile(template_file_quarterly, report_name_quarter)

   
    writer2 = pd.ExcelWriter(report_name_quarter, engine='openpyxl', mode='a', if_sheet_exists = 'overlay')

  
    
    df_months.loc[0].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 4, startcol = 5)
    df_months.loc[2].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 4, startcol = 3)
    df_months.loc[0].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 7, startcol = 3)
    df_months.loc[1].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 7, startcol = 2)
    df_months.loc[2].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 7, startcol = 1)

  
    avg_daily_dfs[0][['lot_no', 'average_load', 'average_aircon', 'average_hotplate']].to_excel(writer2, sheet_name = "firstMonth", index=False, header=False, startrow = 2, startcol = 0)
    
    total_cust_dfs[0][['lot_no','total_load']].to_excel(writer2, sheet_name = "firstMonth", index=False, header=False, startrow = 2, startcol = 6)
    #daily_total_qtr[['Date','daily_total_load']].to_excel(writer2, sheet_name = "Total Daily Use", index=False, header=False, startrow = 1, startcol = 0)
    full_avg_by_lot['lot_no'].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 0)
    full_avg_by_lot.iloc[:,5].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 1)
    full_avg_by_lot.iloc[:,4].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 2)
    full_avg_by_lot.iloc[:,3].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 3)
    full_avg_by_lot['average_load'].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 4)
    avg_daily_dfs[1][['lot_no', 'average_load', 'average_aircon', 'average_hotplate']].to_excel(writer2, sheet_name = "secondMonth", index=False, header=False, startrow = 2, startcol = 0)
    total_cust_dfs[1][['lot_no','total_load']].to_excel(writer2, sheet_name = "secondMonth", index=False, header=False, startrow = 2, startcol = 6)
    
    #daily_lot_avg_dfs[1][['lot_no', 'average_load']].to_excel(writer2, sheet_name = "Q4 average use", index=False, header=False, startrow = 8, startcol = 2)
   
    avg_daily_dfs[2][['lot_no', 'average_load', 'average_aircon', 'average_hotplate']].to_excel(writer2, sheet_name = "thirdMonth", index=False, header=False, startrow = 2, startcol = 0)
    total_cust_dfs[2][['lot_no','total_load']].to_excel(writer2, sheet_name = "thirdMonth", index=False, header=False, startrow = 2, startcol = 6)
    
    #passing the max_tem data from the BOM website
    
    
    final_max_tem[['Date','daily_total_load','Max_Temperature']].to_excel(writer2, sheet_name = "Q4 Total Daily Use", index=False, header=False, startrow = 1, startcol = 0)
    
    writer2.close()

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(report_name_quarter, 'aviva-monthly-reports',report_save_name)
   

