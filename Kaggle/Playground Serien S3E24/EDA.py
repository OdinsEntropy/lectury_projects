# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:43:01 2023

@author: Eric
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:07:36 2023

@author: Eric
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:29:15 2023

@author: Eric
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier,HistGradientBoostingClassifier,BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import IsolationForest

from sklearn.decomposition import PCA

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna
from optuna_dashboard import run_server




#%% Get Input data and split it up 
# TO BEAT AUC 0.79094
# AUC 0.79071 for 100 iter

train_org = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')


#%% AUgmentation

hearing_left_cat = pd.get_dummies(train_org['hearing(left)'], prefix='hearing_left')
hearing_right_cat = pd.get_dummies(train_org['hearing(right)'], prefix='hearing_right')

normal_scaled = StandardScaler().fit_transform(train_org[['height(cm)','weight(kg)','waist(cm)','systolic','relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride','HDL', 'LDL', 'hemoglobin','serum creatinine']])
normal_scaled = pd.DataFrame(normal_scaled, columns=['height(cm)','weight(kg)','waist(cm)','systolic','relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride','HDL', 'LDL', 'hemoglobin','serum creatinine'])

exp_scaled = FunctionTransformer(np.log1p).transform(train_org[['eyesight(left)', 'eyesight(right)', 'Urine protein', 'AST', 'ALT', 'Gtp']])

#train = pd.concat([hearing_left_cat,hearing_right_cat,normal_scaled,exp_scaled], axis=1)

train = pd.concat([normal_scaled, exp_scaled, hearing_left_cat, hearing_right_cat], axis=1)
#train = train_org.drop("smoking", axis=1)


#%% CatBoostClassifier
# 0.8688242888589922

#classifier_obj = CatBoostClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate,task_type="GPU")
classifier_obj = CatBoostClassifier(logging_level='Silent')


auc = cross_validate(classifier_obj, train,train_org.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%% CatBoostClassifier
# 0.8688254857656854

#classifier_obj = CatBoostClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate,task_type="GPU")
classifier_obj = make_pipeline(FunctionTransformer(np.log1p),CatBoostClassifier(logging_level='Silent'))


auc = cross_validate(classifier_obj, train_org.drop('smoking', axis=1),train_org.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%%

models = {'CBC': CatBoostClassifier(**{'task_type'           : "CPU",
                              'objective'           : 'Logloss',
                              'loss_function'       : 'Logloss',
                              'eval_metric'         : 'AUC',
                              'bagging_temperature' : 0.5,
                              'colsample_bylevel'   : 0.5,
                              'iterations'          : 1000,
                              'learning_rate'       : 0.045,
                              'od_wait'             : 45,
                              'max_depth'           : 8,
                              'l2_leaf_reg'         : 1.5,
                              'min_data_in_leaf'    : 30,
                              'random_strength'     : 0.35, 
                              'max_bin'             : 120,
                              'verbose'             : 0,
                              'use_best_model'      : False,
                           }
                         ),
       'LGBM1C': LGBMClassifier(**{'device'           : "cpu",
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


classifier_obj = make_pipeline(PCA(20),models["LGBM1C"])


auc = cross_validate(classifier_obj, train_org.drop('smoking', axis=1),train_org.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

