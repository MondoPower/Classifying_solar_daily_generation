from more_itertools import one
import data_gathering
import pandas as pd
import etl.influx_etl as influx_etl
import os
import numpy as np
import pandas as pd
import pdb
import manipulating_data
import plotly.graph_objs as go
import model_generation
import gather_weather_data
import matplotlib.pyplot as plt


location_id = '0780ec0b-56fa-4242-88ae-0355069045bf'
# location_id = '373f76de-57ac-4618-9d56-f422d6f8b2d7'
daterange = {"start_date": '2022-02-12T00:00:00Z', "end_date": '2022-12-12T00:00:00Z'}
# start_date,end_date = data_gathering.get_date_range(daterange)
# df, location_config = data_gathering.get_location_full(location_id)

# df.to_csv('2022-02_2022-12.csv')
##start_date= '2022-02-12T00:00:00Z'
df = pd.read_csv('2022-02_2022-12.csv')

#df = pd.read_csv('full_data.csv')
df = manipulating_data.manipulate_data(df)
df=manipulating_data.manual_statistics(df)
#df = manipulating_data.statistical_labeling(df)

#data = pd.DataFrame()

# for day in df.date.unique():
    
#     day = '20'+ day + 'T07:00:00Z'
#     df_dayily= gather_weather_data.get_weather_data_from_timestream(start_date=day, horizon_hours = 16)
#     data=pd.concat([data,df_dayily],ignore_index=True)
# data.to_csv('final_big.csv')

final = pd.read_csv('final_big.csv')
#final = pd.read_csv('final.csv')

manipulated_weather = gather_weather_data.manipulate_weather_data(final)
features_all = manipulating_data.merged_data(manipulated_weather,df)
data_normalized, target,features_all = model_generation.scaling_data(features_all)
X_train, y_train,X_test,y_test,dates = model_generation.test_train_split(features_all,data_normalized, target, test_frac=0.15)

predictions = model_generation.LogisticRegression_model(X_train, y_train,X_test,y_test,dates)


  
# for loc in locations.location_id.unique()[0:1]:
#     final_df, pv_training_cols, weather_cols = data_gathering.gather_dataset_for_location(loc)
#     training_data, target = data_wrangling.add_timeseries_cols_and_format(final_df, 'pv')
#     output_models, train_df, test_df, train_time, test_time, pred_test, pred_train  = model_training.main_model_generation(training_data, loc)
#     #model_training.plot_models(test_df, train_df, test_time, train_time, pred_test, pred_train, loc)
#     #model_training.plot_error_models(test_df, train_df, test_time, train_time, pred_test, pred_train, loc)
#     output_df = pd.DataFrame(output_models)
#     metrics = metrics.append(output_df)
#     #print(calc_metrics)
#     filepath = f'{loc} xgb_test.csv'
#     if os.path.exists(filepath) == True:
#          metrics.to_csv(filepath, mode='a', header=False)
#     else:
#          metrics.to_csv(filepath)

