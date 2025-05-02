# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:29:15 2023

@author: Eric
"""

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

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna
from optuna_dashboard import run_server




#%% Get Input data and split it up 
# TO BEAT AUC 0.79094
# AUC 0.79071 for 100 iter

train_org = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')

train = train_org.copy()



#%% SVM-Pipeline

classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                     #Nystroem(n_components=200, random_state=23),
                     #FunctionTransformer(np.log1p),
                     SGDClassifier(loss="log_loss"))
auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])
#%%


# 1. Define an objective function to be maximized.
def objective(trial):

    rf_C = trial.suggest_float('rf_C', 0.0001, 0.003, log=True)
    rf_n_components = trial.suggest_int('rf_n_components', 400, 1000, log=True)


    
    classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                         Nystroem(n_components=rf_n_components, random_state=23),
                         StandardScaler(),
                         SGDClassifier())
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)


#%% RandomForestClassifier

# 1. Define an objective function to be maximized.
def objective(trial):

    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    rf_estimator = trial.suggest_int('rf_estimator', 2, 500, log=True)
    rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])
    rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 32, log=True)
    
    classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_estimator, criterion=rf_criterion, min_samples_leaf=rf_min_samples_split, max_features=1.0, random_state=2, n_jobs=-1)
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return auc

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

#%% RandomForestClassifier-Pipeline

# 1. Define an objective function to be maximized.
def objective(trial):

    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    rf_estimator = trial.suggest_int('rf_estimator', 2, 500, log=True)
    rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])
    rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 32, log=True)

    classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                         RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_estimator, criterion=rf_criterion, min_samples_leaf=rf_min_samples_split, max_features=1.0, random_state=2, n_jobs=-1))
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)


#%% GradientBoostingClassifier

# 1. Define an objective function to be maximized.
def objective(trial):

    rf_learningrate = trial.suggest_float('rf_learningrate', 0.05, 4.0)
    rf_estimator = trial.suggest_int('rf_estimator', 2, 500, log=True)
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    
    classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                                   GradientBoostingClassifier(n_estimators=rf_estimator, learning_rate=rf_learningrate, max_depth=rf_max_depth, random_state=23))
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)


#%% ExtraTreesClassifier

# 1. Define an objective function to be maximized.
def objective(trial):

    rf_estimator = trial.suggest_int('rf_estimator', 2, 500, log=True)
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 200, log=True)
    rf_criterion = trial.suggest_categorical('rf_criterion', ['gini', 'entropy'])


    
    #classifier_obj = make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
    #                     FunctionTransformer(np.log1p),
    #                     ExtraTreesClassifier(n_estimators=rf_estimator,min_samples_leaf=rf_min_samples_split, max_depth=rf_max_depth, criterion=rf_criterion, max_features=1.0,random_state=23,n_jobs=-1))
    
    classifier_obj = ExtraTreesClassifier(n_estimators=rf_estimator,min_samples_leaf=rf_min_samples_split, max_depth=rf_max_depth, criterion=rf_criterion, max_features=1.0,random_state=23,n_jobs=-1) #Not working well
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

#%% LogisticRegression-Pipeline

# 1. Define an objective function to be maximized.
def objective(trial):

    rf_C = trial.suggest_float('rf_C', 0.0001, 0.003, log=True)
    rf_n_components = trial.suggest_int('rf_n_components', 400, 1000, log=True)


    
    classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                         Nystroem(n_components=rf_n_components, random_state=23),
                         StandardScaler(),
                         LogisticRegression(dual=False, C=rf_C, class_weight=None, max_iter=1500,random_state=23,n_jobs=-1,solver='newton-cholesky'))
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

#%% LGBMClassifier-Pipeline
# 1. Define an objective function to be maximized.
def objective(trial):

    rf_estimator = trial.suggest_int('rf_estimator', 100, 1000, log=True)
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    rf_num_leaves = trial.suggest_int('rf_min_samples_split', 2, 20, log=True)
    rf_criterion = trial.suggest_categorical('rf_criterion', ['gbdt', 'dart'])


    
    #classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
    #                     Nystroem(n_components=rf_n_components, random_state=23),
    #                     StandardScaler(),
    #                     LogisticRegression(dual=False, C=rf_C, class_weight=None, max_iter=1500,random_state=23,n_jobs=-1,solver='newton-cholesky'))
    
    classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                     lgb.LGBMClassifier(boosting_type=rf_criterion, n_estimators=rf_estimator, class_weight='balanced', num_leaves=rf_num_leaves, max_depth=rf_max_depth, n_jobs=-1, random_state=23))
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

#%% XGBClassifier-Pipeline
# 1. Define an objective function to be maximized.
def objective(trial):

    rf_estimator = trial.suggest_int('rf_estimator', 100, 1000, log=True)
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
    rf_learningrate = trial.suggest_float('rf_learningrate', 0.05, 4.0)


    
    #classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
    #                     Nystroem(n_components=rf_n_components, random_state=23),
    #                     StandardScaler(),
    #                     LogisticRegression(dual=False, C=rf_C, class_weight=None, max_iter=1500,random_state=23,n_jobs=-1,solver='newton-cholesky'))
    
    classifier_obj = make_pipeline(XGBClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate, n_jobs=-1 ))
    
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

#%% CatBoostClassifier
# 1. Define an objective function to be maximized.
def objective(trial):

    rf_estimator = trial.suggest_int('rf_estimator', 400, 2000, log=True)
    rf_max_depth = trial.suggest_int('rf_max_depth', 1,8)
    rf_learningrate = trial.suggest_float('rf_learningrate', 0.001, 0.3, log=True)

    params={'verbose': False} 

    
    #classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
    #                     Nystroem(n_components=rf_n_components, random_state=23),
    #                     StandardScaler(),
    #                     LogisticRegression(dual=False, C=rf_C, class_weight=None, max_iter=1500,random_state=23,n_jobs=-1,solver='newton-cholesky'))
    
    classifier_obj = CatBoostClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate, logging_level='Silent')
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, scoring='roc_auc',fit_params=params)
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)


#%% Create complete Modell

auc = auc = cross_validate(GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23), train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
result = np.mean(auc["test_score"])

#%% OUTPUT
"""
 RandomForestClassifier-log1ß - ['rf_max_depth': 9, 'rf_estimator': 431, 'rf_criterion': 'entropy', 'rf_min_samples_split': 28, 'rf_n_components': 18]  - 0.7911698727037935
 RandomForestClassifier - [rf_max_depth: 9, rf_estimator: 498, rf_criterion: gini, rf_min_samples_split: 27]                                            - 0.7901235146497249
 GradientBoostingClassifier - [rf_learningrate: 0.19988299934404352, rf_estimator: 24, rf_max_depth: 4]                                                 - 0.7912156893697948
 ExtraTreesClassifier -[rf_estimator: 189, rf_max_depth: 12, rf_min_samples_split: 34, rf_criterion: gini]                                              - 0.7920201547963299
 LGBMClassifier,with log1p- [rf_estimator: 725, rf_max_depth: 3, rf_min_samples_split: 14, rf_criterion: dart]                                          - 0.7924071943076223
 LogisticRegression -  [rf_C: 0.0009095968332141859, rf_n_components: 745, rf_class_weight: balanced]                                                   - 0.7909408133812006
 XGBClassifier -  [rf_estimator: 243, rf_max_depth: 2, rf_learningrate: 0.16643025468333306]                                                            - 0.7922392803323907
 CatBoostClassifier - [rf_estimator: 561, rf_max_depth: 4, rf_learningrate: 0.05823290251374662]                                                        - 0.7920389159175478

"""