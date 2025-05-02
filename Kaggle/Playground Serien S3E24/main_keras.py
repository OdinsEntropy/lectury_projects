# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:14:16 2023

@author: Eric
"""


import datetime

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from sklearn.datasets import make_classification

import optuna
from optuna_dashboard import run_server

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Discretization, BatchNormalization

tf.config.set_visible_devices([], 'GPU') #disable GPU...



#%% Get Input data and split it up 

train = pd.read_csv('input/train.csv', index_col='id')
test = pd.read_csv('input/test.csv', index_col='id')

#%% data augmentation

transf = FunctionTransformer(np.log1p).fit_transform(train.drop('smoking', axis=1))
transf = StandardScaler().fit_transform(transf)

#%%
model = Sequential()
model.add(Dense(96,  activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%%
model = Sequential()
model.add(Dense(5, kernel_initializer='zeros'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])


#%%    


history = model.fit(train.drop('smoking', axis=1),train.smoking, epochs=500, batch_size=500, validation_split=0.3)
#history = model.fit(train.drop('smoking', axis=1),result, epochs=10, batch_size=500, validation_split=0.3,callbacks=[tensorboard_callback])

#auc = cross_val_keras(model, F"Neurons count")

#%% Generate output
#output_pd = pd.DataFrame({'smoking':model.predict(test)[:, 1]}, index=test.index)
#output_pd.to_csv(f"test_pred_{auc}.csv",index=True)


#%%
from lightgbm import LGBMClassifier


models = {       'LGBM1C': LGBMClassifier(**{'device'           : "cpu",
                           'objective'         : 'binary',
                           'metric'            : 'auc',
                           'boosting_type'     : 'gbdt',
                           'random_state'      : 23,
                           'colsample_bytree'  : 0.35,
                           'subsample'         : 0.45,
                           'learning_rate'     : 0.05,
                           'max_depth'         : 8,
                           'n_estimators'      : 1000,
                           'num_leaves'        : 175,                    
                           'reg_alpha'         : 0.01,
                           'reg_lambda'        : 1.75,
                           'verbose'           : -1,
                        }
                     )}


classifier_obj = make_pipeline(models["LGBM1C"])


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%%

models["LGBM1C"].fit(train.drop('smoking', axis=1),train.smoking)

#%%
from xgboost import XGBClassifier


params = {'n_estimators'          : 2048,
          'max_depth'             : 9,
          'learning_rate'         : 0.045,
          'booster'               : 'gbtree',
          'subsample'             : 0.75,
          'colsample_bytree'      : 0.30,
          'reg_lambda'            : 1.00,
          'reg_alpha'             : 0.80,
          'gamma'                 : 0.80,
          'random_state'          : 42,
          'objective'             : 'binary:logistic',
          #'tree_method'           : 'gpu_hist',
          'eval_metric'           : 'auc',
          #'early_stopping_rounds' : 256,
          #'device'                : 'cuda:0',
          'n_jobs'                : -1,
         }

classifier_obj = make_pipeline(XGBClassifier(**params))


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))