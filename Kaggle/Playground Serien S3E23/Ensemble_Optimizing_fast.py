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

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Discretization, BatchNormalization, Normalization
import random
#%%
import itertools

def alle_moeglichen_kombinationen():
    elemente = list(range(6))  # 0 bis 6
    kombinationen = list(itertools.product(elemente, repeat=8))
    kombinationen = [list(kombination) for kombination in kombinationen]
    return kombinationen

# Beispielaufruf:
kombinationen = alle_moeglichen_kombinationen()
print("Anzahl der möglichen Kombinationen:", len(kombinationen))

random_sample= pd.DataFrame()
random_sample["data"] = random.sample(kombinationen, 5000)


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
    

#%%
random_sample["auc"] = 0
for index, row in random_sample.iterrows():
    random_sample.loc[index, 'auc'] = roc_auc_score(train["defects"], np.mean(1/8*new_data * row["data"], axis=1))
    
    
#%%
abc = random_sample.sort_values("auc", axis=0, ascending=False, ignore_index=1).to_numpy()
np.mean(np.vstack(abc[:30,:]), axis=0)