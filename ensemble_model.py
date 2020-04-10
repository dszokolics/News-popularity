from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK

from utils.preprocess import preprocess, preprocess_lm
from utils.helpers import clean_params
import pandas as pd
from tqdm import tqdm
import yaml


# Preprocess data
seed = 1
test_size = 0.2

train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=test_size, pca=False, random_state=seed)
train_p, valid_p, y_var, X_vars_p, test_p, _p = preprocess(test_set=True, test_size=test_size, random_state=seed)
train_lm, valid_lm, test_lm, _lm, X_vars_lm, y_var = preprocess_lm(test_size=test_size, random_state=seed)
h2o.init()

# Load the models
def load(file):
    return yaml.load(open(file, 'r'), Loader=yaml.FullLoader)


rfc_final = RandomForestClassifier(**load('model_params/rfc_01.yaml'))
rfc_final_p = RandomForestClassifier(**load('model_params/rfc_01_p.yaml'))

xgb_final = XGBClassifier(**load('model_params/xgb_01.yaml'))
xgb_final_p = XGBClassifier(**load('model_params/xgb_01_p.yaml'))

dl_final = H2ODeepLearningEstimator(**load('model_params/dl_01.yaml'))
dl_final_p = H2ODeepLearningEstimator(**load('model_params/dl_01_p.yaml'))
dl_final_p2 = H2ODeepLearningEstimator(**load('model_params/dl_02_p.yaml'))

gbm_final = H2OGradientBoostingEstimator(**load('model_params/gbm_02.yaml'))
gbm_final_p = H2OGradientBoostingEstimator(**load('model_params/gbm_01_p.yaml'))
gbm_final2 = H2OGradientBoostingEstimator(**load('model_params/gbm_01.yaml'))

lm_final = H2OGeneralizedLinearEstimator(**load('model_params/lm_01.yaml'))

# Create dictionaries and sets of models in order to iterate over them
sklearn_estimators = {'rfc': rfc_final, 'rfc_p': rfc_final_p, 'xgb': xgb_final, 'xgb_p': xgb_final_p}
h2o_estimators = {'dl': dl_final, 'dl_p': dl_final_p, 'dl_p2': dl_final_p2, 'gbm': gbm_final, 'gbm2': gbm_final2, 'gbm_p': gbm_final_p}
early_stop = {'xgb', 'xgb_p', 'gbm', 'gbm2', 'gbm_p', 'dl', 'dl_p', 'dl_p2'}

# Train sklearn models
for name, est in tqdm(sklearn_estimators.items()):

    train_set = train_p if name.endswith('_p') else train
    valid_set = valid_p if name.endswith('_p') else valid
    X = X_vars_p if name.endswith('_p') else X_vars

    if name in early_stop:
        params = {'early_stopping_rounds': 8, 'eval_metric': 'logloss',
                  'eval_set': [(valid_set[X], valid_set[y_var])]}
    else:
        params = {}

    # Saving cross validation predictions to use them for the ensemble model
    cross_val_preds = cross_val_predict(est, train_set[X], train_set[y_var],
                                        method='predict_proba',
                                        fit_params=params)
    train[f'{name}_pred'] = [x[1] for x in cross_val_preds]
    est.fit(train_set[X], train_set[y_var], **params)

# Create H2OFrames from dataframes
train_h2o = h2o.H2OFrame(train)
train_h2o[y_var] = train_h2o[y_var].asfactor()
valid_h2o = h2o.H2OFrame(valid)
valid_h2o[y_var] = valid_h2o[y_var].asfactor()
train_p_h2o = h2o.H2OFrame(train_p)
train_p_h2o[y_var] = train_p_h2o[y_var].asfactor()
valid_p_h2o = h2o.H2OFrame(valid_p)
valid_p_h2o[y_var] = valid_p_h2o[y_var].asfactor()

# Train H2O models & save cross validation predictions for ensemble modeling
for name, est in tqdm(h2o_estimators.items()):

    train_set = train_p_h2o if name.endswith('_p') else train_h2o
    valid_set = valid_p_h2o if name.endswith('_p') else valid_h2o
    X = X_vars_p if name.endswith('_p') else X_vars

    validation_frame = valid_set if name in early_stop else None
    est.train(x=X, y=y_var, training_frame=train_set, validation_frame=validation_frame)
    res = est.cross_validation_predictions()
    res = [x.as_data_frame() for x in res]
    res = pd.concat(res, axis=1)
    train[f'{name}_pred'] = res['p1'].sum(axis=1).values

# Train the linear model
train_lm_h2o = h2o.H2OFrame(train_lm)
train_lm_h2o[y_var] = train_lm_h2o[y_var].asfactor()
valid_lm_h2o = h2o.H2OFrame(valid_lm)
valid_lm_h2o[y_var] = valid_lm_h2o[y_var].asfactor()

lm_final.train(x=X_vars_lm, y=y_var, training_frame=train_lm_h2o, validation_frame=valid_lm_h2o)
res = lm_final.cross_validation_predictions()
res = [x.as_data_frame() for x in res]
res = pd.concat(res, axis=1)
train[f'lm_pred'] = res['p1'].sum(axis=1).values

# Declare prediction columns
pred_cols = [x for x in train.columns if x.endswith('pred')]
train[pred_cols]

# Check the correlation of predictions
train[pred_cols].corr()

# Create predictions for a given dataset
def predict(df, df_p, df_lm, sklearn_estimators, h2o_estimators, meta, X_vars, X_vars_p, X_vars_lm, pred_cols):
    for name, est in sklearn_estimators.items():
        if name.endswith('_p'):
            df[f'{name}_pred'] = [x[1] for x in est.predict_proba(df_p[X_vars_p])]
        else:
            df[f'{name}_pred'] = [x[1] for x in est.predict_proba(df[X_vars])]

    df_h2o = h2o.H2OFrame(df[X_vars])
    df_p_h2o = h2o.H2OFrame(df_p[X_vars_p])
    df_lm_h2o = h2o.H2OFrame(df_lm[X_vars_lm])
    for name, est in h2o_estimators.items():
        if name.endswith('_p'):
            res = est.predict(df_p_h2o)
            res = res.as_data_frame()
        else:
            res = est.predict(df_h2o)
            res = res.as_data_frame()
        df[f'{name}_pred'] = res.p1.values

    if df_lm is not None:
        res = lm_final.predict(df_lm_h2o)
        res = res.as_data_frame()
        df['lm_pred'] = res.p1.values

    df['final_pred'] = meta.predict(h2o.H2OFrame(df[pred_cols])).as_data_frame().values

    return df


# Get base learner scores
def base_scores(df, pred_cols):
    for col in pred_cols:
        print(f'{col}: {roc_auc_score(df[y_var], df[col])}')


base_scores(train, pred_cols)

# Train the meta learner
t, v = train_test_split(train, test_size=0.2)
t_h2o = h2o.H2OFrame(t)
v_h2o = h2o.H2OFrame(v)

# Hyperparameter optimization
def hyperopt_train_test(params):
    lm = H2OGeneralizedLinearEstimator(**params)
    lm.train(x=pred_cols, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)
    return lm.model_performance(v_h2o).rmse()

space = {
    'Lambda': hp.quniform('Lambda', 0.001, 1, 0.001),
    'alpha': hp.quniform('alpha', 0, 1, 0.01),
    'family': 'gaussian',
    'early_stopping': True,
    'seed': 1
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)

best

# Save params
params = clean_params(best)
params['family'] = 'gaussian'
params['early_stopping'] = True
params['seed'] = 1

yaml.dump(params, open('model_params/meta.yaml', 'w'), indent=0)

# Create & train meta learner model with the best parameters
meta = H2OGeneralizedLinearEstimator(**params)
meta.train(x=pred_cols, y=y_var, training_frame=t_h2o, validation_frame=v_h2o)

# Check prediction performance on the holdout set
_ = predict(_, _p, _lm, sklearn_estimators, h2o_estimators, meta, X_vars, X_vars_p, X_vars_lm, pred_cols)
roc_auc_score(_[y_var], _['final_pred'])

# Get final predictions for the test set
test = predict(test, test_p, test_lm, sklearn_estimators, h2o_estimators, meta, X_vars, X_vars_p, X_vars_lm, pred_cols)

test.rename(columns={'final_pred': 'score'}, inplace=True)
result = test[['article_id', 'score']].copy()

result.score.hist(bins=20)
# result.to_csv('results/s20_ens8_lm.csv', index=False)
