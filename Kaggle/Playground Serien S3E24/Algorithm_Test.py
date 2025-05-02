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
# 0.8387910193388091

classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                     #Nystroem(n_components=200, random_state=23),
                     #FunctionTransformer(np.log1p),
                     SGDClassifier(loss="log_loss"))
auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% RandomForestClassifier
# 0.8575407309156782

#classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_estimator, criterion=rf_criterion, min_samples_leaf=rf_min_samples_split, max_features=1.0, random_state=2, n_jobs=-1)
classifier_obj = RandomForestClassifier(random_state=23, n_jobs=-1)

auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% GradientBoostingClassifier
# 0.8586037156810871

#classifier_obj = classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
#                               GradientBoostingClassifier(n_estimators=rf_estimator, learning_rate=rf_learningrate, max_depth=rf_max_depth, random_state=23))

classifier_obj = classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                               GradientBoostingClassifier(random_state=23))

auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% ExtraTreesClassifier
# 0.8528490754317157

#classifier_obj = ExtraTreesClassifier(n_estimators=rf_estimator,min_samples_leaf=rf_min_samples_split, max_depth=rf_max_depth, criterion=rf_criterion, max_features=1.0,random_state=23,n_jobs=-1) #Not working well
classifier_obj = ExtraTreesClassifier(random_state=23,n_jobs=-1) #Not working well


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% LogisticRegression-Pipeline
# 0.8511148501836576

"""
classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                     Nystroem(n_components=rf_n_components, random_state=23),
                     StandardScaler(),
                     LogisticRegression(dual=False, C=rf_C, class_weight=None, max_iter=1500,random_state=23,n_jobs=-1,solver='newton-cholesky'))
"""


classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                     Nystroem(n_components=500, random_state=23),
                     StandardScaler(),
                     LogisticRegression(max_iter=500, random_state=23,n_jobs=-1))

auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%% LGBMClassifier-Pipeline
# 

#classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
#                 lgb.LGBMClassifier(boosting_type=rf_criterion, n_estimators=rf_estimator, class_weight='balanced', num_leaves=rf_num_leaves, max_depth=rf_max_depth, n_jobs=-1, random_state=23))

classifier_obj = make_pipeline(FunctionTransformer(np.log1p),
                 lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1, random_state=23))

auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% XGBClassifier-Pipeline
# 0.864707530090762

#classifier_obj = make_pipeline(XGBClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate, n_jobs=-1 ))

classifier_obj = make_pipeline(XGBClassifier(n_jobs=-1 ))


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
np.mean(auc["test_score"])

#%% CatBoostClassifier
# 0.8688242888589922

#classifier_obj = CatBoostClassifier(n_estimators=rf_estimator, max_depth=rf_max_depth, learning_rate=rf_learningrate,task_type="GPU")
classifier_obj = CatBoostClassifier(logging_level='Silent')


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%% Ensemble
abc=[('RF', make_pipeline(FunctionTransformer(np.log1p),
                              RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),
    
    ('et', make_pipeline(FunctionTransformer(np.log1p),
                         ExtraTreesClassifier(n_estimators=189,min_samples_leaf=34, criterion='gini', max_depth=12,  max_features=1.0,random_state=23,n_jobs=-1))),
    
    ('hgb', HistGradientBoostingClassifier(random_state=23)),
    
    ('ny', make_pipeline(FunctionTransformer(np.log1p),
                         Nystroem(n_components=745, random_state=23),
                         StandardScaler(),
                         LogisticRegression(dual=False, C=0.0009095968332141859, max_iter=1500, class_weight='balanced', random_state=23,n_jobs=-1))),
    
    ('GBC', GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23)),
    
    #('LGBM', make_pipeline(FunctionTransformer(np.log1p),
    #                       lgb.LGBMClassifier(boosting_type='dart', n_estimators=725, class_weight='balanced', num_leaves=5, max_depth=3, n_jobs=-1, random_state=23))),
     
    ('XGB', XGBClassifier(n_estimators=243, max_depth=2, learning_rate=0.16643025468333306, n_jobs=-1, random_state=23)),
    
    ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23,logging_level='Silent'))
    ]

score = []
for classifier in abc:
    
    auc = cross_validate(classifier[1], train.drop('smoking', axis=1),train.smoking, cv=5, scoring='roc_auc', verbose=3)
    print(np.mean(auc["test_score"]))
    score.append([classifier[0], np.mean(auc["test_score"])])


#%% Ensemble
# 0.861795082864613

ensemble = VotingClassifier(
    [
     ('RF', make_pipeline(RandomForestClassifier(random_state=23, n_jobs=-1))),

     ('et', make_pipeline(FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=500, random_state=23),
                          StandardScaler(),
                          LogisticRegression(class_weight='balanced', random_state=23,n_jobs=-1))),
     
     ('GBC', GradientBoostingClassifier(random_state=23)),
     
     ('LGBM', make_pipeline(FunctionTransformer(np.log1p),
                            lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1, random_state=23))),
      
     ('XGB', XGBClassifier(n_jobs=-1, random_state=23)),
     
     ('CATB', CatBoostClassifier(random_seed=23,logging_level='Silent'))
     ], voting='soft')


auc = cross_validate(ensemble, train.drop('smoking', axis=1),train.smoking, cv=5, scoring=['roc_auc'], verbose=3)
result = np.mean(auc["test_roc_auc"])

#%%
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Discretization, BatchNormalization
from scikeras.wrappers import KerasClassifier

#%%
def create_model():

    model = Sequential()
    model.add(Dense(512, input_dim=22, activation='relu'))
    model.add(Dropout(0.4))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    return model



#%%

classifier_obj = KerasClassifier(model=create_model, epochs=100, batch_size=500, validation_split=0.3)


classifier_obj = make_pipeline(StandardScaler(),
                               KerasClassifier(model=create_model, epochs=100, batch_size=500, validation_split=0.3))

#FunctionTransformer(np.log1p),


if 1:
    classifier_obj.fit(train.drop('smoking', axis=1),train.smoking)

else:
    auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, scoring=['roc_auc'], verbose=3)
    result = np.mean(auc["test_roc_auc"])
    
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


#auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
#print(np.mean(auc["test_score"]))




#%% Generate


classifier_obj.fit(train.drop('smoking', axis=1),train.smoking)
output_pd = pd.DataFrame({'smoking':classifier_obj.predict_proba(test)[:, 1]}, index=test.index)

Version = "2"
try:
    output_pd.to_csv(f"output/submission_v{Version}.csv",index=True, mode='x')
    
    with open("output/#meta.txt", "a") as text_file:
        print(f"V{Version}      {str(classifier_obj)}       {str(classifier_obj.get_params())}", file=text_file)
        
except FileExistsError:
    print("ERROR FILE EXIST")
