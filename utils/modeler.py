import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


class ModelResults():

    def __init__(self, archive_path=None):
        self._results = []
        self._archive_path = archive_path

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        self._results = results

    def clear_results(self, archive=True, archive_path=None):
        if archive:
            self.archive(archive_path)
        self._results = []

    @property
    def archive_path(self):
        return self._archive_path

    @archive_path.setter
    def archive_path(self, path):
        self._archive_path = path

    def store_results(self, model, feat, result):
        d = {'ts': dt.datetime.now(),
             'type': model.__class__,
             'params': model.get_params(),
             'feat': feat,
             'results': result}
        if isinstance(model, RandomForestClassifier):
            d['params']['n_features'] = model.n_features_
        self._results.append(d)

    def archive(self, path=None):
        path = self._archive_path if path is None else path
        pickle.dump(self._results,
                    open(f'{path}_results_{dt.datetime.now()}', 'wb'))

        print(f'{len(self._results)} results archived.')

    def eval_models(self, params, std=False):

        res = []
        for result in self._results:

            d = {'type': result['type']}

            if (issubclass(result['type'], LinearRegression)
                | issubclass(result['type'], DecisionTreeRegressor)
                | issubclass(result['type'], RandomForestRegressor)
                | issubclass(result['type'], GradientBoostingRegressor)
                | issubclass(result['type'], Lasso)
                | issubclass(result['type'], Ridge)
                | issubclass(result['type'], ElasticNet)):
                metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']
            else:
                metrics = ['train_auc', 'test_auc']

            for metric in metrics:
                d[metric] = np.mean([x[metric] for x in result['results']])
                if std:
                    d[f'std_{metric}'] = np.std([x[metric] for x in result['results']])

            for param in params:
                try:
                    d[param] = result['params'][param]
                except KeyError:
                    d[param] = None

            res.append(d)

        res = pd.DataFrame(res) if len(res) > 1 else pd.DataFrame(res, index=[0])

        return res

    def eval_features(self):

        res = []
        id = 0
        for result in self._results:
            metric = []
            for subresult in result['results']:
                if 'fi' in subresult.keys():
                    type = 'Feature importance'
                    metric.append(subresult['fi'])
                elif 'coef' in subresult.keys():
                    type = 'Coefficient'
                    metric.append(subresult['coef'])

            metric = pd.DataFrame(metric)
            metric.columns = result['feat']
            metric = metric.transpose()
            metric['mn'] = metric.mean(axis=1)

            metric.sort_values('mn', ascending=False, inplace=True)
            metric['type'] = type
            metric['model_id'] = id

            res.append(metric)
            id +=1

        res = pd.concat(res)

        return res


class KFoldModeler():

    def __init__(self, n_splits, model=None, binary=False):
        self.kf = KFold(n_splits, random_state=20200203, shuffle=True)
        self._model = model
        self._results = ModelResults('results/')
        self._binary = binary

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def set_model_param(self, params):
        self._model.set_params(**params)

    @property
    def results(self):
        return self._results._results

    @results.setter
    def results(self, results):
        self._results._results = results

    @property
    def binary(self):
        return self._binary

    @binary.setter
    def binary(self, value):
        self._binary = value

    def archive_results(self):
        self._results.archive()

    def run(self, X, y, threshold=None):
        res = []

        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self._model.fit(X_train, y_train)
            res.append(self.model_kpi(X_train, X_test, y_train, y_test, threshold))
            # fi.append(rfc.feature_importances_)

        self._results.store_results(self._model, X.columns.tolist(), res)

        return res

    def model_kpi(self, X_train, X_test, y_train, y_test, threshold):

        y_pred_test = self._model.predict(X_test)
        y_pred_train = self._model.predict(X_train)
        if self._binary:
            y_pred_test = y_pred_test.clip(0, 1)
            y_pred_train = y_pred_train.clip(0, 1)
            ret_dict = {'train_auc': roc_auc_score(y_train, y_pred_train),
                        'test_auc': roc_auc_score(y_test, y_pred_test)}
        else:
            ret_dict = {'train_rmse': mean_squared_error(y_train, y_pred_train, squared=False),
                        'test_rmse': mean_squared_error(y_test, y_pred_test, squared=False),
                        'train_r2': self._model.score(X_train, y_train),
                        'test_r2': self._model.score(X_test, y_test)}

        if (isinstance(self._model, RandomForestClassifier)
            | isinstance(self._model, DecisionTreeRegressor)
            | isinstance(self._model, RandomForestRegressor)
            | isinstance(self._model, GradientBoostingRegressor)):
            ret_dict['fi'] = self._model.feature_importances_

        elif (isinstance(self._model, LogisticRegression)):
            ret_dict['coef'] = self._model.coef_[0]

        elif (isinstance(self._model, LinearRegression)
              | isinstance(self._model, Lasso)
              | isinstance(self._model, Ridge)
              | isinstance(self._model, ElasticNet)):
            ret_dict['coef'] = self._model.coef_

        if threshold is not None:
            y_pred_test = y_pred_test > threshold
            conf_mx = pd.DataFrame({'y_real': y_test, 'y_pred': y_pred_test})
            conf_mx.reset_index(inplace=True)
            conf_mx = conf_mx.groupby(['y_real', 'y_pred']).comp_id.count()
            ret_dict['confusion_matrix'] = conf_mx

        return ret_dict

    def params_search(self, X, y, params):
        res = {}
        for p, v in params.items():
            original = self._model.get_params()[p]

            for value in tqdm(v):
                self.set_model_param({p: value})
                res[f'{p}_{str(value)}'] = self.run(X, y)

            self.set_model_param({p: original})

        return res

    def grid_search(self, X, y, params):
        if len(params) == 1:
            self.params_search(X, y, params)
        else:
            param = params.popitem()
            original = self._model.get_params()[param[0]]

            for p in param[1]:
                self.set_model_param({param[0]: p})
                self.grid_search(X, y, params)

            self.set_model_param({param[0]: original})

    def eval(self, what='models', params=[], std=False):
        if what == 'models':
            return self._results.eval_models(params, std)
        elif what == 'features':
            return self._results.eval_features()
