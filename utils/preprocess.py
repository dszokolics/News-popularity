import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class GroupPCA():

    def __init__(self):
        self.pcas = {}

    def fit(self, df, groups):
        i = 0
        for g in groups:
            pca = PCA(len(g))
            pca.fit(df[g])

            self.pcas[f'group{i}'] = {'columns': g, 'pca': pca}
            i+=1

    def transform(self, df):
        df = df.copy()
        dfs = []
        for k, p in self.pcas.items():
            pca_df = pd.DataFrame(p['pca'].transform(df[p['columns']]))
            pca_df.rename(columns={x: f'{k}_comp{x}' for x in pca_df.columns}, inplace=True)
            pca_df.index = df.index
            dfs.append(pca_df)

            df.drop(columns=p['columns'], inplace=True)

        df = pd.concat([df]+dfs, axis=1)

        return df

    def fit_transform(self, df, groups):
        self.fit(df, groups)
        return self.transform(df)


def logarithmize(df, X_vars):
    cols_to_log = [x for x in X_vars if (x.startswith('self_reference') | x.startswith('LDA')
                                         | (x in ['kw_max_min', 'kw_min_max', 'kw_min_avg']))]
    for col in cols_to_log:
        df[col] = np.log(df[col]).fillna(-1).clip(-1, 100)

    return df


def preprocess(test_set=False, test_size=0.5, pca=True, random_state=1):

    random_state = random_state

    train = pd.read_csv('data/train.csv')
    if test_set:
        train, _ = train_test_split(train, test_size=0.1, random_state=random_state)

    train.set_index('article_id', inplace=True)
    train.drop(columns='is_weekend', inplace=True)

    y_var = 'is_popular'
    pre_PCA_X_vars = [x for x in train.columns if x != y_var]

    train, valid = train_test_split(train, test_size=test_size, random_state=random_state)

    train = logarithmize(train, pre_PCA_X_vars)

    scaler = StandardScaler()
    train.loc[:, pre_PCA_X_vars] = scaler.fit_transform(train[pre_PCA_X_vars])

    if pca:

        corr = train[pre_PCA_X_vars].corr().stack().reset_index()
        corr = corr[corr.level_0 > corr.level_1].sort_values(0, ascending=False)
        # corr.head(10)
        c2 = corr[corr[0].abs() > 0.7].copy()
        groups = []
        while len(c2) > 0:
            col = c2.level_0.values[0]
            c3 = c2[(c2.level_0 == col) | (c2.level_0 == col)]
            group = set(c3.level_0.tolist()) | set(c3.level_1.tolist())
            c2 = c2[~(c2.level_0.isin(group)) & ~(c2.level_1.isin(group))]

            groups.append(group)

        gpca = GroupPCA()
        train = gpca.fit_transform(train, groups)

        X_vars = [x for x in train.columns if x != y_var]

    else:
        X_vars = pre_PCA_X_vars

    valid = logarithmize(valid, pre_PCA_X_vars)
    valid.loc[:, pre_PCA_X_vars] = scaler.transform(valid[pre_PCA_X_vars])
    if pca: valid = gpca.transform(valid)

    test = pd.read_csv('data/test.csv')
    test = logarithmize(test, pre_PCA_X_vars)
    test.loc[:, pre_PCA_X_vars] = scaler.transform(test[pre_PCA_X_vars])
    if pca: test = gpca.transform(test)

    if test_set:
        _ = logarithmize(_, pre_PCA_X_vars)
        _.loc[:, pre_PCA_X_vars] = scaler.transform(_[pre_PCA_X_vars])
        if pca: _ = gpca.transform(_)

        return train, valid, y_var, X_vars, test, _

    else:
        return train, valid, y_var, X_vars, test


def preprocess_lm(test_size, random_state):
    train, valid, y_var, X_vars, test, _ = preprocess(test_set=True, test_size=test_size, random_state=random_state, pca=False)
    # train2, valid2, x, x, x, _2 = preprocess(test_set=True, test_size=0.2, pca=False)

    bools = {'data_channel_is_', 'weekday_is_'}
    to_drop = []
    for b in bools:
        cols = [x for x in train.columns if x.startswith(b)]
        to_drop.append(train[cols].sum().sort_values(ascending=False).index.values[0])
    train.drop(columns=to_drop, inplace=True)
    valid.drop(columns=to_drop, inplace=True)
    test.drop(columns=to_drop, inplace=True)
    _.drop(columns=to_drop, inplace=True)
    # train2.drop(columns=to_drop, inplace=True)
    # valid2.drop(columns=to_drop, inplace=True)
    # _2.drop(columns=to_drop, inplace=True)

    corr = train.corr().stack().reset_index()
    corr = corr[corr.level_0 > corr.level_1].sort_values(0, ascending=False)
    corr.head(10)
    c2 = corr[corr[0].abs() > 0.7].copy()
    groups = []
    while len(c2) > 0:
        col = c2.level_0.values[0]
        c3 = c2[(c2.level_0 == col) | (c2.level_0 == col)]
        group = set(c3.level_0.tolist()) | set(c3.level_1.tolist())
        c2 = c2[~(c2.level_0.isin(group)) & ~(c2.level_1.isin(group))]

        groups.append(group)

    gpca = GroupPCA()
    train = gpca.fit_transform(train, groups)
    valid = gpca.transform(valid)
    test = gpca.transform(test)
    _ = gpca.transform(_)
    # train2 = gpca.transform(train2)
    # valid2 = gpca.transform(valid2)
    # _2 = gpca.transform(_2)

    corr = train.corr().stack().reset_index()
    corr = corr[corr.level_0 > corr.level_1].sort_values(0, ascending=False)
    corr

    train.drop(columns='self_reference_min_shares', inplace=True)
    test.drop(columns='self_reference_min_shares', inplace=True)
    valid.drop(columns='self_reference_min_shares', inplace=True)
    _.drop(columns='self_reference_min_shares', inplace=True)
    # train2.drop(columns='self_reference_min_shares', inplace=True)
    # valid2.drop(columns='self_reference_min_shares', inplace=True)
    # _2.drop(columns='self_reference_min_shares', inplace=True)

    X_vars = [x for x in train.columns if x != y_var]

    cont = train[X_vars].nunique()
    cont = cont[cont > 10].index.tolist()

    def feature_eng(df, cont):
        df = df.copy()
        for c in cont:
            df[f'{c}_squared'] = df[c] ** 2
            df[f'{c}_root'] = df[c].abs() ** 0.5 * np.sign(df[c])
            df[f'{c}_log'] = (np.log(df[c].abs()) * np.sign(df[c])).clip(0, 100).fillna(-1)

        return df

    train = feature_eng(train, cont)
    valid = feature_eng(valid, cont)
    test = feature_eng(test, cont)
    _ = feature_eng(_, cont)
    # train2 = feature_eng(train2, cont)
    # valid2 = feature_eng(valid2, cont)
    # _2 = feature_eng(_2, cont)

    assert train.isna().sum().sum() == 0

    X_vars = [x for x in train.columns if x != y_var]

    train.corr()[y_var].sort_values(ascending=False).iloc[1:10]

    to_interact = ['group4_comp0_root', 'LDA_03_root', 'num_imgs_root', 'global_subjectivity_root', 'num_hrefs_root']

    def interact(df, to_interact, X_vars):
        df = df.copy()
        for i in range(0, len(to_interact)):
            for col in [x for x in X_vars if x not in [to_interact[i:]]]:
                df[f'{to_interact[i]}_x_{col}'] = df[to_interact[i]] * df[col]

        return df

    train = interact(train, to_interact, X_vars)
    valid = interact(valid, to_interact, X_vars)
    test = interact(test, to_interact, X_vars)
    _ = interact(_, to_interact, X_vars)
    # train2 = interact(train2, to_interact, X_vars)
    # valid2 = interact(valid2, to_interact, X_vars)
    # _2 = interact(_2, to_interact, X_vars)

    X_vars = [x for x in train.columns if x != y_var]

    scaler2 = StandardScaler()
    train.loc[:, X_vars] = scaler2.fit_transform(train[X_vars])
    valid.loc[:, X_vars] = scaler2.transform(valid[X_vars])
    test.loc[:, X_vars] = scaler2.transform(test[X_vars])
    _.loc[:, X_vars] = scaler2.transform(_[X_vars])
    # train2.loc[:, X_vars] = scaler2.fit_transform(train2[X_vars])
    # valid2.loc[:, X_vars] = scaler2.transform(valid2[X_vars])
    # _2.loc[:, X_vars] = scaler2.transform(_2[X_vars])

    return train, valid, test, _, X_vars, y_var
