from more_itertools import one
import data_gathering
import pandas as pd
import etl.influx_etl as influx_etl
import os
import numpy as np
import pandas as pd
import pdb
import manipulating_data



location_id = '0780ec0b-56fa-4242-88ae-0355069045bf'
daterange = {"start_date": '2022-08-19T00:00:00Z', "end_date": '2022-10-19T00:00:00Z'}
start_date,end_date = data_gathering.get_date_range(daterange)
df, location_config = data_gathering.get_location_full(location_id)
df = manipulating_data.manipulate_data(df)
df_high_low= manipulating_data.return_high_low_solar(df)
weather = pd.read_csv('solcast_data.csv',delimiter='\t')
merged_data=data_gathering.manipulate_weather_data(weather,df_high_low)


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

