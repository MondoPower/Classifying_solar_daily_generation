from more_itertools import one
import data_gathering
import pandas as pd
import etl.influx_etl as influx_etl
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import pdb
import manipulating_data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def scaling_data(features_all):

    cols_to_use = [c for c in features_all.columns if 'dates' not in c and 'date' not in c and 'pv_labeled' not in c]
    data = features_all[cols_to_use].copy()
    target = features_all['pv_labeled']
    scaler = StandardScaler()
    scaler.fit(data)

    data_normalized = scaler.transform(data)
    data_normalized = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)

    print(data_normalized.describe().round(2))

    return data_normalized,target

def test_train_split(data_normalized, target,test_frac=0.15):
    import pdb;pdb.set_trace()
    data_normalized.reset_index(drop=True, inplace=True)
    split_loc = int(np.floor(len(data_normalized)*(1-test_frac)))

    
    x_train = data_normalized.loc[0:split_loc]
    x_test = data_normalized.loc[split_loc:]
    y_train = target.loc[0:split_loc]
    y_test = target.loc[split_loc:]
    
    return x_train, y_train,x_test,y_test


def LogisticRegression_model(x_train, y_train,x_test,y_test):
    import pdb;pdb.set_trace()
    
    model =  LogisticRegression(random_state=0)
    model.fit(x_train,y_train)
    print(model.score(x_test, y_test))
    y_predictions = model.predict(x_test)
    print(classification_report(y_test, y_predictions))
  
    # do coef, class weights in this model if you balance 
    #balance in train, which column is important and windowing the time
    

    return predictions
    

  


