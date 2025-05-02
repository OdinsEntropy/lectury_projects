import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import Nystroem

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import IsolationForest

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna
from optuna_dashboard import run_server

#%% Get Input data and split it up 

train = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')
original = pd.read_csv('./input/jm1.csv',na_values=['?'])


#%% Copy 0.79267
# 0.7923250575109038 with weights weights=[0.3, 0.3, 0.1, 0.3]
# 0.7922865164265807 no weights


ensemble = VotingClassifier(
    [
     ('RF', RandomForestClassifier( max_depth=9, n_estimators=498, criterion='gini', min_samples_leaf=27, max_features=1.0, random_state=23, n_jobs=-1)),

     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=100,min_samples_leaf=100, max_features=1.0,random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=400, random_state=23),
                          StandardScaler(),
                          LogisticRegression(dual=False, C=0.0032,max_iter=1500,random_state=23,n_jobs=-1))),
     
     ],
    
    voting='soft')#,weights=[0.3, 0.3, 0.1, 0.3])

auc = cross_validate(ensemble, train.drop('defects', axis=1),train.defects, cv=5, scoring=['roc_auc'], n_jobs=-1)
result = np.mean(auc["test_roc_auc"])


#%% Optimized

# 0.7926884255709838 no weights
# 0.7926896833794693 [0.07,0.15,0.15,0.07,0.11,0.15,0.15,0.15]
# 0.7926970551541792 [1,2,2,1,1,2,2,2]
# 0.7926285456108679 [2,1,1,2,2,1,1,1]
# 0.7927537869463743 [1,2,1,1,1,3,2,1]
# 0.7927844082432284 [1,3,1,1,1,3,3,1]
# 0.7927916825675531 [1,5,1,1,1,5,5,1]

# 0.7927931285890414 [0,5,1,1,1,5,5,1]

# 0.7927688318292315 [0,5,0,0,1,5,5,1]
# 0.7927468068893216 [0,5,1,0,1,5,5,1]

#optimized
# 0.7927885448287888 [4,3,2,3,4,4,4,4]


# 0.7928276616667113 [3,5,0,2,0,4,4,5]

ensemble = VotingClassifier(
    [
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
     
     ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23))
     ], voting='soft', weights=[3,5,0,2,0,4,4,5])


auc = cross_validate(ensemble, train.drop('defects', axis=1),train.defects, cv=5, scoring=['roc_auc'])
result = np.mean(auc["test_roc_auc"])


#%%

def objective(trial):

    rf_1 = trial.suggest_int('rf_1', 0, 5)
    rf_2 = trial.suggest_int('rf_2', 0, 5)
    rf_3 = trial.suggest_int('rf_3', 0, 5)
    rf_4 = trial.suggest_int('rf_4', 0, 5)
    rf_5 = trial.suggest_int('rf_5', 0, 5)
    rf_6 = trial.suggest_int('rf_6', 0, 5)
    rf_7 = trial.suggest_int('rf_7', 0, 5)
    rf_8 = trial.suggest_int('rf_8', 0, 5)



    
    classifier_obj = ensemble = VotingClassifier(
        [
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
         
         ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23, logging_level='Silent'))
         ], voting='soft', weights=[rf_1,rf_2,rf_3,rf_4,rf_5,rf_6,rf_7,rf_8])
    
    auc = cross_validate(classifier_obj, train.drop('defects', axis=1),train.defects, cv=5, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=100)

#run_server(storage)

#%%

def objective(trial):

    rf_1 = trial.suggest_int('rf_1', 0, 8)
    rf_2 = trial.suggest_int('rf_2', 0, 8)

    classifier_obj = ensemble = VotingClassifier(
        [
         ('RF', make_pipeline(FunctionTransformer(np.log1p),
                              RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),
         
         ('hgb', HistGradientBoostingClassifier(random_state=23)),

         ], voting='soft', weights=[rf_1,rf_2])
    
    auc = cross_validate(classifier_obj, train.drop('defects', axis=1),train.defects, cv=5, n_jobs=-1, scoring='roc_auc')
    return np.mean(auc["test_score"])

# 3. Create a study object and optimize the objective function.
#study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3")
#study.optimize(objective, n_trials=10)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(direction='maximize', storage=storage)
study.optimize(objective, n_trials=2)

#run_server(storage)



#%%
ensemble.fit(train.drop('defects', axis=1),train.defects)
output_pd = pd.DataFrame({'defects':ensemble.predict_proba(test)[:, 1]}, index=test.index)
output_pd.to_csv(f"test_pred_ensemble_opt_8ens_{result}.csv",index=True)

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