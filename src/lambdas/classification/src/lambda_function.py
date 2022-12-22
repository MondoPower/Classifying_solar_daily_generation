from more_itertools import one
import libs.data_gathering as data_gathering
import libs.manipulating_data as  manipulating_data
import pandas as pd
import etl.influx_etl as influx_etl

def handler(event, context):
    """
    This is the lambda entry point
    Event contains inputs for the lambda
    context contains things about how it was called like AWS role.
    """

    """
    Steps to fill in:
    1. parse event into relevant input variables
    2. run etl/cleaning/data_wrangling
    3. Run model training and get testing metrics
    4. Run final model training
    5. Return final model, metadata, metric info that would go to a database in the future.
    """
   
    community_name = event['community_name']
    report_type= event['report_type']

   #  print(report_type)
    if report_type=='quarterly':
       manipulating_data.quarterly_report(community_name)
    else:
       manipulating_data.write_monthly_report(community_name)
        
    return
