import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from preprocess import preprocess, preprocess_lm

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


train, valid, test, _, X_vars, y_var = preprocess_lm(test_size=0.5, random_state=1)

lm = LogisticRegression(max_iter=3000)
lm.fit(train[X_vars], train[y_var])

roc_auc_score(train[y_var], lm.predict(train[X_vars]))
roc_auc_score(valid[y_var], lm.predict(valid[X_vars]))

h2o.init(min_mem_size='4048M')

t_h2o = h2o.H2OFrame(train)
v_h2o = h2o.H2OFrame(valid)
test_h2o = h2o.H2OFrame(test)
_h2o = h2o.H2OFrame(_)
# t2_h2o = h2o.H2OFrame(train2)
# v2_h2o = h2o.H2OFrame(valid2)
# _2_h2o = h2o.H2OFrame(_2)

t_h2o[y_var] = t_h2o[y_var].asfactor()
v_h2o[y_var] = v_h2o[y_var].asfactor()
_h2o[y_var] = _h2o[y_var].asfactor()
# t2_h2o[y_var] = t2_h2o[y_var].asfactor()
# v2_h2o[y_var] = v2_h2o[y_var].asfactor()
# _2_h2o[y_var] = _2_h2o[y_var].asfactor()


def hyperopt_train_test(params):
    lm = H2OGeneralizedLinearEstimator(**params)
    lm.train(x=X_vars, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)
    return lm.model_performance(v_h2o).logloss()

space = {
    'Lambda': hp.quniform('Lambda', 0.001, 1, 0.001),  # hp.choice('Lambda', [2**x for x in range(-10, 11)]),
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

for key, val in best.items():
    if isinstance(best[key], np.int32):
        best[key] = int(val)
    elif isinstance(best[key], np.float):
        best[key] = float(round(val, 7))

params = best.copy()
params['family'] = 'binomial'
params['early_stopping'] = True
params['seed'] = 1
params['keep_cross_validation_predictions'] = True
params['nfolds'] = 5

params

yaml.dump(params, open('models/lm_01.yaml', 'w'), indent=0)

lm_final = H2OGeneralizedLinearEstimator(**params)

# lm_final.train(x=X_vars, y=y_var, training_frame=t2_h2o, validation_frame=v2_h2o)
# res = lm_final.cross_validation_predictions()
# res = [x.as_data_frame() for x in res]
# res = pd.concat(res, axis=1)

# train2[f'lm_pred'] = res['p1'].sum(axis=1).values
# train2[['lm_pred']].to_csv('models/lm_01_train.csv')


# train2.shape

lm_final.train(x=X_vars, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)
lm_final.auc()

# res = lm_final.predict(_2_h2o)
# res = res.as_data_frame()
# _2['pred'] = res.p1.values

# roc_auc_score(_2[y_var], _2['pred'])

# _result = _2[['article_id', 'pred']].copy()
# _result.rename(columns={'pred': 'score'}, inplace=True)

# _result[['article_id', 'score']].to_csv('models/lm_01_t.csv', index=False)
# _result.article_id.sort_values()

score = lm_final.predict(test_h2o)
score = score.as_data_frame()
test['score'] = score.p1.values

# test[['article_id', 'score']].to_csv('results/s16_lm.csv', index=False)
