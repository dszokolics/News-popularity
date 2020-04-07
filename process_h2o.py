import pandas as pd
from pandas_profiling import ProfileReport
from tqdm import tqdm
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import xgboost


train = pd.read_csv('data/train.csv')
train.shape
train.set_index('article_id', inplace=True)

test = pd.read_csv('data/test.csv')
test.shape
test.set_index('article_id', inplace=True)

y_var = 'is_popular'
X_vars = [x for x in train.columns if x != y_var]

train[y_var] = train[y_var].astype(bool)

h2o.init()

train_h = h2o.H2OFrame(train)
test_h = h2o.H2OFrame(test)

train_h, valid_h, _ = train_h.split_frame(ratios=[.75, .249], seed=20200322)

x = X_vars
y = y_var

aml = H2OAutoML(max_runtime_secs=300, seed=2)
aml.train(x=x, y=y, training_frame=train_h, validation_frame=valid_h)
aml.leaderboard

preds = aml.predict(test_h)

preds = preds.as_data_frame()
preds.head()

preds['True'].hist(bins=20)

preds = pd.concat([test.reset_index(), preds], axis=1)
preds.reset_index(inplace=True)

preds.rename(columns={'True': 'score'}, inplace=True)
preds = preds[['article_id', 'score']]

preds.shape
preds.to_csv('results/s2_automl.csv', index=False)
