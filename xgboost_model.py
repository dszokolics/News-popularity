import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess
import pickle

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=0.2, pca=True)

def hyperopt_train_test(params):
    params['max_depth'] = int(params['max_depth'])
    xgb = XGBClassifier(**params)
    xgb.fit(train[X_vars], train[y_var], early_stopping_rounds=8, eval_metric='logloss',
            eval_set=[(train[X_vars], train[y_var]), (valid[X_vars], valid[y_var])])
    return xgb.evals_result()['validation_1']['logloss'][-8]

space = {
    'max_depth': hp.quniform('max_depth', 2, 7, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.01),
    'subsample': hp.quniform('subsample', 0.3, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 1.3, 0.01),
    'reg_alpha': hp.quniform('reg_alpha', 0, 18, 1),
    'reg_lambda': hp.quniform('reg_lambda', 1, 2.3, 0.1),
    'seed': 21,
    'n_estimators': 500,
    'eta': hp.quniform('eta', 0.01, 0.3, 0.01),
    'objective': 'binary:logistic'
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=30, trials=trials)

best

params = best.copy()
params['max_depth'] = int(params['max_depth'])
params['n_estimators'] = 500
params['objective'] = 'binary:logistic'
params['seed'] = 21

xgb_fin0 = XGBClassifier(**best)
xgb_fin0.max_depth = xgb_fin0.max_depth + 1
xgb_fin0.n_estimators = 500
xgb_fin0.objective = 'binary:logistic'
xgb_fin0.seed = 21

fit_params = {'early_stopping_rounds': 8, 'eval_metric': 'logloss',
              'eval_set': [(valid[X_vars], valid[y_var])]}
cross_validate(xgb_fin0, train[X_vars], train[y_var],
               scoring='roc_auc', return_train_score=True, fit_params=fit_params)

xgb_fin0.fit(train[X_vars], train[y_var], **fit_params)

roc_auc_score(_[y_var], [x[1] for x in xgb_fin0.predict_proba(_[X_vars])])

pickle.dump(params, open('models/xgb_01_p.pckl', 'wb'))
#
# xgb = pickle.load(open('models/xgb_01.pckl', 'rb'))
