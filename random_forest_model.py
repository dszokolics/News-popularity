import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from preprocess import preprocess
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=0.5, pca=True)

def hyperopt_train_test(params):
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, train[X_vars], train[y_var], scoring='neg_log_loss', cv=4).mean()

space = {
    'max_depth': hp.quniform('max_depth', 10, 50, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 40, 1),
    'n_estimators': 400,
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'min_impurity_decrease': 1e-7,
    'max_samples': hp.quniform('max_samples', 0.3, 0.99, 0.01),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'random_state': 12
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)

best

params = best.copy()
params['n_estimators'] = 400
params['max_features'] = ['sqrt', 'log2'][params['max_features']]
params['min_impurity_decrease'] = 1e-7
params['criterion'] = ['gini', 'entropy'][params['criterion']]
params['random_state'] = 12
params['min_samples_split'] = int(params['min_samples_split'])
params['max_depth'] = int(params['max_depth'])

rfc_final = RandomForestClassifier(**params)

cross_validate(rfc_final, train[X_vars], train[y_var], scoring='roc_auc', return_train_score=True)

rfc_final.fit(train[X_vars], train[y_var])
roc_auc_score(valid[y_var], [x[1] for x in rfc_final.predict_proba(valid[X_vars])])

pickle.dump(params, open('models/rfc_01_p.pckl', 'wb'))

# test['score'] = [x[1] for x in rfc_final.predict_proba(test[X_vars])]
#
# result = test[['article_id', 'score']].copy()
#
# result.score.hist(bins=25)
# result.to_csv('results/s08_rf.csv', index=False)
