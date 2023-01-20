from more_itertools import one
import data_gathering
import pandas as pd
import etl.influx_etl as influx_etl
import os
import numpy as np
import pandas as pd
import pdb
import manipulating_data
from sklearn.linear_model import LogisticRegression

def LogisticRegression_model(merged_df):

    model =  LogisticRegression(random_state=0)
    copy_data = merged_df.copy()
    target = merged_df.pop('pv_labeled')
    

    pdb.set_trace()
    # Fit and plot
    model.fit(merged_df,target)
    # plot_classifier(X,y,model,proba=True)

    # Predict probabilities on training points
    prob = model.predict_proba(X)
    print("Maximum predicted probability", np.max(prob))

    return predictions



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

    

  


