from more_itertools import one
import data_gathering
import pandas as pd
import etl.influx_etl as influx_etl
from sklearn.preprocessing import StandardScaler
import os
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import pdb
import sklearn
import manipulating_data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def scaling_data(features_all):
       
    data = features_all.copy()
    data.drop(columns=['labeled_target','date'],inplace=True)
    
    target = features_all['labeled_target']
    scaler = StandardScaler()
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    data_normalized = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)

    return data_normalized,target

def test_train_split(features_all,data_normalized, target,test_frac=0.15):
    
    #regular random split, check the season difference ,train last week, and test on few days
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, random_state=42)
    dates = features_all.date.copy()
    # data_normalized.reset_index(drop=True, inplace=True)
    # split_loc = int(np.floor(len(data_normalized)*(1-test_frac)))
    # x_train = data_normalized.loc[0:split_loc]
    # x_test = data_normalized.loc[split_loc:]
    # y_train = target.loc[0:split_loc]
    # y_test = target.loc[split_loc:]
    return X_train, y_train,X_test,y_test,dates


def LogisticRegression_model(X_train, y_train,X_test,y_test,dates):
    import pdb;pdb.set_trace()
    logistic_Model = LogisticRegression(random_state=1234)
    # elastic_net_classifier =  LogisticRegressionCV(class_weight='balanced',cv=4, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9],solver='saga')
    logistic_Model.fit(X_train,y_train)
    y_predictions = logistic_Model.predict(X_test)
  
    y_test.reset_index(drop=True)
    times =dates.iloc[0:91]
    #times=dates.iloc[137:162]
    new_compare = pd.concat([times,y_test],axis=1)
    new_compare['predictions'] =y_predictions
    new_compare.rename(columns={'labeled_target':'actuals'},inplace=True)
    print(classification_report(y_test, y_predictions))
   
    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=new_compare['date'], y=new_compare['actuals'],mode='lines', name='actuals', line=dict(color='red')))
    # fig.add_trace(go.Scattergl(x=new_compare['date'], y=new_compare['predictions'], mode='lines', name='predicted', line=dict(color='blue',dash='dash')))
    # fig.update_yaxes(title_text='new location 11-1and 0.5Max')
    # fig.show()

    # coef = elastic_net_classifier.coef_
    # intercept = elastic_net_classifier.intercept_
    
    return y_predictions

   
    
    

  


