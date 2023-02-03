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
from sklearn.linear_model import Ridge

def scaling_data(features_all):

    
    data = features_all.copy()
    data.drop(columns=['labeled_target'],inplace=True)
    target = features_all['labeled_target']
    scaler = StandardScaler()
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    data_normalized = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)

    print(data_normalized.describe().round(2))

    return data_normalized,target

def test_train_split(data_normalized, target,test_frac=0.15):
 
    data_normalized.reset_index(drop=True, inplace=True)
    split_loc = int(np.floor(len(data_normalized)*(1-test_frac)))
    x_train = data_normalized.loc[0:split_loc]
    x_test = data_normalized.loc[split_loc:]
    y_train = target.loc[0:split_loc]
    y_test = target.loc[split_loc:]
    
    return x_train, y_train,x_test,y_test


def LogisticRegression_model(x_train, y_train,x_test,y_test):
    
    elastic_net_classifier =  LogisticRegressionCV(class_weight='balanced',cv=4, penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9],solver='saga')
    import pdb;pdb.set_trace()
    elastic_net_classifier.fit(x_train,y_train)
    
    y_predictions = elastic_net_classifier.predict(x_test)
    print(classification_report(y_test, y_predictions))
    coef = elastic_net_classifier.coef_
    intercept = elastic_net_classifier.intercept_
    # Set up the parameter grid
    param_grid = {"alpha": np.linspace(0.00001, 1, 10),'solver':['sag','Lsqr']}
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    ridge=Ridge()
    Tuned lasso paramaters: {'alpha': 0.11112, 'solver': 'sag'}
    # Instantiate lasso_cv
    ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)

    # Fit to the training data
    ridge_cv.fit(X_train, y_train)
    print("Tuned lasso paramaters: {}".format(ridge_cv.best_params_))
    print("Tuned lasso score: {}".format(ridge_cv.best_score_))
    
    return y_predictions


 #and windowing the time,feature engineering
   
    
    

  


