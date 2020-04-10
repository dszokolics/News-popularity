from utils.preprocess import preprocess
from utils.helpers import clean_params

import pandas as pd
import numpy as np
import yaml

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.metrics import roc_auc_score
import h2o


### Preprocessing

train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=0.15, pca=False)

h2o.init()

t = h2o.H2OFrame(train)
v = h2o.H2OFrame(valid)
_p = h2o.H2OFrame(_)

t[y_var] = t[y_var].asfactor()
v[y_var] = v[y_var].asfactor()
_p[y_var] = _p[y_var].asfactor()

### Hyperparameter optimization

def hyperopt_train_test(params):
    p = params.copy()
    for key in ['max_depth', 'min_rows', 'nbins']:
        p[key] = int(p[key])
    dl = H2OGradientBoostingEstimator(**p)
    dl.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
    return dl.model_performance(v).logloss()

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
    'stopping_rounds': 5,
    'stopping_metric': 'logloss',
    'stopping_tolerance': 1e-4,
    'nfolds': 5,
    'seed': 1
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)

best

### Test & save params

params = clean_params(best)
for key in ['max_depth', 'min_rows', 'nbins']:
    params[key] = int(params[key])

params['distribution'] = 'multinomial'
params['min_split_improvement'] = [1e-10, 1e-8, 1e-6, 1e-4][params['min_split_improvement']]
params['stopping_rounds'] = 5
params['stopping_metric'] = 'logloss'
params['stopping_tolerance'] = 1e-4
params['nfolds'] = 5
params['seed'] = 1
params['ntrees'] = 500
params['keep_cross_validation_predictions'] = True

gbm_final = H2OGradientBoostingEstimator(**params)


gbm_final.train(x=X_vars, y=y_var, training_frame=t, validation_frame=v)
gbm_final.auc(xval=True)

res = gbm_final.predict(_p)
res = res.as_data_frame()
_['pred'] = res.p1.values

roc_auc_score(_[y_var], _['pred'])

params

yaml.dump(params, open('model_params/gbm_02.yaml', 'w'))

# test_h2o = h2o.H2OFrame(test)
# res = gbm_final.predict(test_h2o)
# res = res.as_data_frame()
# test['score'] = res.p1.values
#
# result = test[['article_id', 'score']].copy()
# result.score.hist(bins=20)
#
# result.to_csv('results/s11_gbm.csv', index=False)
