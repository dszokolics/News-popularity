import pandas as pd
import numpy as np
import yaml

from utils.preprocess import preprocess, preprocess_lm
from utils.helpers import clean_params

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


### Data preprocessing

train, valid, test, _, X_vars, y_var = preprocess_lm(test_size=0.5, random_state=1)

# Check the potential with an unrestricted logistic regression
lm = LogisticRegression(max_iter=3000)
lm.fit(train[X_vars], train[y_var])

# I can't have better scores after penalization than train scores without penalty.
roc_auc_score(train[y_var], lm.predict(train[X_vars]))

h2o.init(min_mem_size='4048M')

t_h2o = h2o.H2OFrame(train)
v_h2o = h2o.H2OFrame(valid)
test_h2o = h2o.H2OFrame(test)
_h2o = h2o.H2OFrame(_)

t_h2o[y_var] = t_h2o[y_var].asfactor()
v_h2o[y_var] = v_h2o[y_var].asfactor()
_h2o[y_var] = _h2o[y_var].asfactor()


### Hyperparameter optimization

def hyperopt_train_test(params):
    lm = H2OGeneralizedLinearEstimator(**params)
    lm.train(x=X_vars, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)
    return lm.model_performance(v_h2o).logloss()

space = {
    'Lambda': hp.quniform('Lambda', 0.001, 1, 0.001),
    'alpha': hp.quniform('alpha', 0, 1, 0.01),
    'family': 'binomial',
    'early_stopping': True,
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
params['family'] = 'binomial'
params['early_stopping'] = True
params['seed'] = 1
params['keep_cross_validation_predictions'] = True
params['nfolds'] = 5

lm_final = H2OGeneralizedLinearEstimator(**params)

lm_final.train(x=X_vars, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)
res = lm_final.cross_validation_predictions()
res = [x.as_data_frame() for x in res]
res = pd.concat(res, axis=1)

lm_final.auc()

yaml.dump(params, open('model_params/lm_01.yaml', 'w'), indent=0)
