from utils.preprocess import preprocess

import pandas as pd
import numpy as np
import yaml

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import h2o
from sklearn.metrics import roc_auc_score


# Broader non-PCA hidden layers + higher dropout

train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=0.15, pca=True)

h2o.init()

t = h2o.H2OFrame(train)
v = h2o.H2OFrame(valid)
_p = h2o.H2OFrame(_)

t[y_var] = t[y_var].asfactor()
v[y_var] = v[y_var].asfactor()
_p[y_var] = _p[y_var].asfactor()

def hyperopt_train_test(params):
    dl = H2ODeepLearningEstimator(**params)
    if 'hidden' in params.keys():
        dl.hidden = list(params['hidden'])
    if 'hidden_dropout_ratios' in params.keys():
        dl.hidden_dropout_ratios = list(params['hidden_dropout_ratios'])
    dl.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
    return dl.model_performance(v).logloss()

space = {
    'activation': 'rectifierwithdropout',
    'l1': hp.quniform('l1', 0, 1e-3, 1e-6),
    'l2': hp.quniform('l2', 0, 0.1, 1e-3),
    'hidden': hp.choice('hidden', [[10, 10], [10, 20], [10, 30], [10, 50],
                                   [20, 10], [20, 20], [20, 30], [20, 50],
                                   [30, 10], [30, 20],
                                   [50, 10], [50, 20]]),
    'hidden_dropout_ratios': hp.choice('hidden_dropout_ratios', [[0, 0], [0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4],
                                                                [0.1, 0], [0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4],
                                                                [0.2, 0], [0.2, 0.1], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4],
                                                                [0.3, 0], [0.3, 0.1], [0.3, 0.2], [0.3, 0.3], [0.3, 0.4],
                                                                [0.4, 0], [0.4, 0.1], [0.4, 0.2]]),
    # 'hidden': hp.choice('hidden', [[10]*x for x in range(1, 6)]),
    'epochs': hp.choice('epochs', range(6, 20)),
    'input_dropout_ratio': hp.quniform('input_dropout_ratio', 0, 0.7, 0.01),
    'stopping_rounds': 4,
    'stopping_metric': 'logloss',
    'stopping_tolerance': 1e-5,
    'seed': 1
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)

best

for key, val in best.items():
     if isinstance(best[key], np.int32):
         best[key] = int(val)

params = best.copy()
params['hidden'] = [[10, 10], [10, 20], [10, 30], [10, 50],
                   [20, 10], [20, 20], [20, 30], [20, 50],
                   [30, 10], [30, 20],
                   [50, 10], [50, 20]][params['hidden']]
params['hidden_dropout_ratios'] = [[0, 0], [0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4],
                                [0.1, 0], [0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4],
                                [0.2, 0], [0.2, 0.1], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4],
                                [0.3, 0], [0.3, 0.1], [0.3, 0.2], [0.3, 0.3], [0.3, 0.4],
                                [0.4, 0], [0.4, 0.1], [0.4, 0.2]][params['hidden_dropout_ratios']]
params['epochs'] += 5
params['stopping_rounds'] = 4
params['stopping_metric'] = 'logloss'
params['stopping_tolerance'] = 1e-5
params['seed'] = 1
params['activation'] = 'rectifierwithdropout'

dl_final = H2ODeepLearningEstimator(**params)

dl_final.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
dl_final.auc()

res = dl_final.predict(_p)
res = res.as_data_frame()
_['pred'] = res.p1.values

roc_auc_score(_[y_var], _['pred'])

yaml.dump(params, open('model_params/dl_01_p.yaml', 'w'), indent=0)
