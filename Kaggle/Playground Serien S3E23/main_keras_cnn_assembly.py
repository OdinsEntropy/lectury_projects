# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:38:23 2023

@author: Eric
"""

"""
@author: Eric
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier,HistGradientBoostingClassifier,BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scikeras.wrappers import KerasClassifier


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Discretization, BatchNormalization, Normalization

#%% Get Input data and split it up 

train = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')
original = pd.read_csv('./input/jm1.csv',na_values=['?'])

#%%
clf = [
     ('RF', make_pipeline(FunctionTransformer(np.log1p),
                          RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),

     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=189,min_samples_leaf=34, criterion='gini', max_depth=12,  max_features=1.0,random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=745, random_state=23),
                          StandardScaler(),
                          LogisticRegression(dual=False, C=0.0009095968332141859, max_iter=1500, class_weight='balanced', random_state=23,n_jobs=-1))),
     
     ('GBC', GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23)),
     
     ('LGBM', make_pipeline(FunctionTransformer(np.log1p),
                            lgb.LGBMClassifier(boosting_type='dart', n_estimators=725, class_weight='balanced', num_leaves=5, max_depth=3, n_jobs=-1, random_state=23))),
      
     ('XGB', XGBClassifier(n_estimators=243, max_depth=2, learning_rate=0.16643025468333306, n_jobs=-1, random_state=23)),
     
     ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23,task_type="GPU"))]
    
trained= []
for i in clf:
#    trained.append(i[1].fit(train.drop('defects', axis=1),train.defects))
    trained.append(i[1].fit(train.drop('defects', axis=1),train.defects))
    
    
#%% Data augmentation

new_data = pd.DataFrame()
for i, j in zip(clf,trained):
    new_data[i[0]] = j.predict_proba(train.drop('defects', axis=1))[:,1]
    
new_data_test = pd.DataFrame()
for i, j in zip(clf,trained):
    new_data_test[i[0]] = j.predict_proba(test)[:,1]
    
    
    
#%% CNN Model Best one with only percentages
def create_model():

    model = Sequential()
    model.add(Dense(4, input_dim=8,activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    return model



#%%    
#history = model.fit(new_data,y_train, epochs=500, batch_size=500, validation_split=0.3)
history = model.fit(new_data,train.defects, epochs=12, batch_size=500)


#%%

classifier_obj = LogisticRegression(random_state=23,n_jobs=-1)
model = KerasClassifier(model=create_model, epochs=12, batch_size=500, verbose=0)

auc = cross_validate(classifier_obj, new_data,train.defects, cv=5, scoring='roc_auc')
print(np.mean(auc["test_score"]))





#%%
output_pd = pd.DataFrame({'defects':model.predict(new_data_test).squeeze()}, index=test.index)
#%%
output_pd.to_csv(f"test_ensemble_weight_neural.csv",index=True)