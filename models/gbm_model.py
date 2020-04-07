from utils.preprocess import preprocess

import pandas as pd
import numpy as np
import yaml

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.metrics import roc_auc_score
import h2o

train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=0.15)

h2o.init()

t = h2o.H2OFrame(train)
v = h2o.H2OFrame(valid)
_p = h2o.H2OFrame(_)

t[y_var] = t[y_var].asfactor()
v[y_var] = v[y_var].asfactor()
_p[y_var] = _p[y_var].asfactor()

def hyperopt_train_test(params):
    p = params.copy()
    for key in ['max_depth', 'min_rows', 'nbins']:
        p[key] = int(p[key])
    dl = H2OGradientBoostingEstimator(**p)
    dl.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
    return dl.auc(xval=True)

space = {
    'ntrees': 500,
    'max_depth': hp.quniform('max_depth', 2, 7, 1),
    'min_rows': hp.quniform('min_rows', 1, 40, 1),
    'nbins': hp.quniform('nbins', 10, 100, 1),
    'learn_rate': hp.quniform('learn_rate', 0.01, 0.3, 0.01),
    'distribution': 'multinomial',
    'sample_rate': hp.quniform('sample_rate', 0.2, 1, 0.01),
    'col_sample_rate': hp.quniform('col_sample_rate', 0.2, 1, 0.01),
    'min_split_improvement': hp.choice('min_split_improvement', [1e-10, 1e-8, 1e-6, 1e-4]),
    'stopping_rounds': 8,
    'stopping_metric': 'AUC',
    'stopping_tolerance': 1e-5,
    'nfolds': 5,
    'seed': 1
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=30, trials=trials)

best

for key, val in best.items():
     if isinstance(best[key], np.int32):
         best[key] = int(val)

for key in ['max_depth', 'min_rows', 'nbins']:
    best[key] = int(best[key])

params = best.copy()
params['distribution'] = 'multinomial'
params['min_split_improvement'] = [1e-10, 1e-8, 1e-6, 1e-4][params['min_split_improvement']]
params['stopping_rounds'] = 8
params['stopping_metric'] = 'AUC'
params['stopping_tolerance'] = 1e-6
params['nfolds'] = 5
params['seed'] = 1
params['ntrees'] = 500

gbm_final = H2OGradientBoostingEstimator(**params)


gbm_final.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
gbm_final.auc(xval=True)

res = gbm_final.predict(_p)
res = res.as_data_frame()
_['pred'] = res.p1.values

roc_auc_score(_[y_var], _['pred'])

yaml.dump(params, open('model_params/gbm_01_p.yaml', 'w'))

# test_h2o = h2o.H2OFrame(test)
# res = gbm_final.predict(test_h2o)
# res = res.as_data_frame()
# test['score'] = res.p1.values
#
# result = test[['article_id', 'score']].copy()
# result.score.hist(bins=20)
#
# result.to_csv('results/s11_gbm.csv', index=False)
