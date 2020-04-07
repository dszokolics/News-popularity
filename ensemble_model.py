from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.preprocess import preprocess, preprocess_lm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import h2o
import pickle
from tqdm import tqdm
import yaml


seed = 2
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

sklearn_estimators = {'rfc': rfc_final, 'rfc_p': rfc_final_p, 'xgb': xgb_final, 'xgb_p': xgb_final_p}
h2o_estimators = {'dl': dl_final, 'dl_p': dl_final_p, 'dl_p2': dl_final_p2, 'gbm': gbm_final, 'gbm2': gbm_final2, 'gbm_p': gbm_final_p}
early_stop = {'xgb', 'xgb_p', 'gbm', 'gbm2', 'gbm_p', 'dl', 'dl_p', 'dl_p2'}

for name, est in tqdm(sklearn_estimators.items()):

    train_set = train_p if name.endswith('_p') else train
    valid_set = valid_p if name.endswith('_p') else valid
    X = X_vars_p if name.endswith('_p') else X_vars

    if name in early_stop:
        params = {'early_stopping_rounds': 8, 'eval_metric': 'logloss',
                  'eval_set': [(valid_set[X], valid_set[y_var])]}
    else:
        params = {}

    cross_val_preds = cross_val_predict(est, train_set[X], train_set[y_var],
                                        method='predict_proba',
                                        fit_params=params)
    train[f'{name}_pred'] = [x[1] for x in cross_val_preds]
    est.fit(train_set[X], train_set[y_var], **params)

train_h2o = h2o.H2OFrame(train)
train_h2o[y_var] = train_h2o[y_var].asfactor()
valid_h2o = h2o.H2OFrame(valid)
valid_h2o[y_var] = valid_h2o[y_var].asfactor()
train_p_h2o = h2o.H2OFrame(train_p)
train_p_h2o[y_var] = train_p_h2o[y_var].asfactor()
valid_p_h2o = h2o.H2OFrame(valid_p)
valid_p_h2o[y_var] = valid_p_h2o[y_var].asfactor()

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

train_lm_h2o = h2o.H2OFrame(train_lm)
train_lm_h2o[y_var] = train_lm_h2o[y_var].asfactor()
valid_lm_h2o = h2o.H2OFrame(valid_lm)
valid_lm_h2o[y_var] = valid_lm_h2o[y_var].asfactor()

lm_final.train(x=X_vars_lm, y=y_var, training_frame=train_lm_h2o, validation_frame=valid_lm_h2o)
res = lm_final.cross_validation_predictions()
res = [x.as_data_frame() for x in res]
res = pd.concat(res, axis=1)
train[f'lm_pred'] = res['p1'].sum(axis=1).values

pred_cols = [x for x in train.columns if x.endswith('pred')]
train[pred_cols]

train[pred_cols].corr()

def predict(df, df_p, sklearn_estimators, h2o_estimators, meta, X_vars, X_vars_p, pred_cols):
    for name, est in sklearn_estimators.items():
        if name.endswith('_p'):
            df[f'{name}_pred'] = [x[1] for x in est.predict_proba(df_p[X_vars_p])]
        else:
            df[f'{name}_pred'] = [x[1] for x in est.predict_proba(df[X_vars])]

    df_h2o = h2o.H2OFrame(df[X_vars])
    df_p_h2o = h2o.H2OFrame(df_p[X_vars_p])
    for name, est in h2o_estimators.items():
        if name.endswith('_p'):
            res = est.predict(df_p_h2o)
            res = res.as_data_frame()
        else:
            res = est.predict(df_h2o)
            res = res.as_data_frame()
        df[f'{name}_pred'] = res.p1.values

    df['final_pred'] = [x[1] for x in meta.predict_proba(df[pred_cols])]

    return df


def base_scores(df, pred_cols):
    for col in pred_cols:
        print(f'{col}: {roc_auc_score(df[y_var], df[col])}')

base_scores(train, pred_cols)

pred_cols = [x for x in pred_cols if x not in {'lm_pred'}]  #, 'rfc_p_pred', 'gbm_pred'}]

lm = LogisticRegression(max_iter=2000, penalty='l2', C=0.175)
param_grid = {'C': [.15, .175, .2]}
grid = GridSearchCV(lm, param_grid=param_grid, cv=5, scoring='neg_log_loss')
grid.fit(train[pred_cols], train[y_var])
pd.DataFrame(grid.cv_results_)

cross_val_score(lm, train[pred_cols], train[y_var], scoring='neg_log_loss').mean()
lm.fit(train[pred_cols], train[y_var])
log_loss(train[y_var], [x[1] for x in lm.predict_proba(train[pred_cols])])
res = pd.DataFrame({f: c for f, c in zip(pred_cols, lm.coef_[0])}, index=[0]).transpose().sort_values(0)

res

_lm = pd.read_csv('models/lm_01_t.csv')
_ = pd.merge(_, _lm, how='left', on='article_id')
_.rename(columns={'score': 'lm_pred'}, inplace=True)

_ = predict(_, _p, sklearn_estimators, h2o_estimators, lm, X_vars, X_vars_p, pred_cols)

base_scores(_, pred_cols)
roc_auc_score(_[y_var], _['final_pred'])
roc_auc_score(_[y_var], _['final_pred'])

test = predict(test, test_p, sklearn_estimators, h2o_estimators, lm, X_vars, X_vars_p, pred_cols)

test.rename(columns={'final_pred': 'score'}, inplace=True)
result = test[['article_id', 'score']].copy()

result.score.hist(bins=20)
result.to_csv('results/s18_ens8.csv', index=False)
