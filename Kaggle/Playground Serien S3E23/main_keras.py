# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:14:16 2023

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

train = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')
original = pd.read_csv('./input/jm1.csv',na_values=['?'])

#%% data augmentation

transf = FunctionTransformer(np.log1p).fit_transform(train.drop('defects', axis=1))
transf = StandardScaler().fit_transform(transf)

#%%
model = Sequential()
model.add(Dense(96,  activation='relu'))
model.add(Dense(64,  activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%%
model = Sequential()
model.add(Dense(64,  activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(32,  activation='relu'))
model.add(Dense(32,  activation='relu'))
model.add(Dense(16,  activation='relu'))
model.add(Dense(16,  activation='relu'))
model.add(Dense(4,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%%
model = Sequential()
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%%    
history = model.fit(transf,train.defects, epochs=500, batch_size=500, validation_split=0.3)

#auc = cross_val_keras(model, F"Neurons count")

#%% Generate output
#output_pd = pd.DataFrame({'defects':model.predict(test)[:, 1]}, index=test.index)
#output_pd.to_csv(f"test_pred_{auc}.csv",index=True)
