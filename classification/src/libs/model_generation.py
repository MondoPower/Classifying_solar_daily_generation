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
    #mask = features_all[(features_all['date'] >='22-06-01') & (features_all['date'] <= '22-08-30')]
    
    data = features_all.copy()
    data.drop(columns=['labeled_target','date'],inplace=True)
    
    target = features_all['labeled_target']
    scaler = StandardScaler()
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    data_normalized = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)

    return data_normalized,target,features_all


def test_train_split(features_all,data_normalized, target,test_frac=0.15):
   
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, random_state=42)
    dates = features_all.date.copy()

    # #sampling less data
    # X_train = data_normalized.loc[0:149]
    # X_test = data_normalized.loc[149:]
    # y_train = target.loc[0:149]
    # y_test = target.loc[149:]

    # data_normalized.reset_index(drop=True, inplace=True)
    # split_loc = int(np.floor(len(data_normalized)*(1-test_frac)))
    # x_train = data_normalized.loc[0:split_loc]
    # x_test = data_normalized.loc[split_loc:]
    # y_train = target.loc[0:split_loc]
    # y_test = target.loc[split_loc:]
    return X_train, y_train,X_test,y_test,dates


def LogisticRegression_model(X_train, y_train,X_test,y_test,dates):

    logistic_Model = LogisticRegression(random_state=1234)
    #elastic_net_classifier =  LogisticRegressionCV(class_weight='balanced',cv=4, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9],solver='saga')
    logistic_Model.fit(X_train,y_train)
    y_predictions = logistic_Model.predict(X_test)
    import pdb;pdb.set_trace()
    predicted=[1., 1., 0., 0., 1., 1., 0., 0.]
    new = pd.DataFrame(y_test)
    y_test['predictions'] =predicted
    #make dataframe for less data
    # df.rename(columns={'labeled_target':'actuals'},inplace=True)
    # #times =dates.iloc[0:41]
    # y_test['predictions'] =y_predictions
    # new_compare = pd.concat([times,y_test],axis=1)
    # new_compare.fillna(0,inplace=True)
    # compare =new_compare.iloc[14:20]
    # new_compare.reset_index(drop=True)
    # compare['predictions'] =y_predictions
    # compare.rename(columns={'labeled_target':'actuals'},inplace=True)
    print(classification_report(y_test, y_predictions))
   
    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=df.index, y=df['actuals'],mode='lines', name='actuals', line=dict(color='red')))
    # fig.add_trace(go.Scattergl(x=df.index, y=df['predictions'], mode='lines', name='predicted', line=dict(color='blue',dash='dash')))
    # fig.update_yaxes(title_text='new location 11-1and 0.5Max')
    # fig.show()

    # coef = elastic_net_classifier.coef_
    # intercept = elastic_net_classifier.intercept_
    
    return y_predictions

   
    
    

  


