import pandas as pd


class Cleaner(object):
    """An object which does the cleaning. It can >learn< the patterns from the
    train set and apply it to the validation set."""

    def __init__(self):
        self._missing_col_threshold = 50
        self._filter_cols = []
        self._bin_cols = []
        self._clip_cols = []
        self._drop_cols = set()
        self._onehot_cols = set()
        self._fillmin_cols = set()
        self._fillmean_cols = set()
        self._fillzero_cols = set()

    @property
    def drop_cols(self):
        return self._drop_cols

    @drop_cols.setter
    def drop_cols(self, cols):
        self._drop_cols = cols

    @property
    def onehot_cols(self):
        return self._onehot_cols

    @onehot_cols.setter
    def onehot_cols(self, cols):
        self._onehot_cols = cols

    @property
    def fillmin_cols(self):
        return self._fillmin_cols

    @fillmin_cols.setter
    def fillmin_cols(self, cols):
        if set(cols) & (self._fillmean_cols | self._fillzero_cols):
            raise ValueError('Column fillna mode already defined')
        else:
            self._fillmin_cols = cols

    @property
    def fillmean_cols(self):
        return self._fillmean_cols

    @fillmean_cols.setter
    def fillmean_cols(self, cols):
        if set(cols) & (self._fillmin_cols | self._fillzero_cols):
            raise ValueError('Column fillna mode already defined')
        else:
            self._fillmean_cols = cols

    @property
    def fillzero_cols(self):
        return self._fillzero_cols

    @fillzero_cols.setter
    def fillzero_cols(self, cols):
        if set(cols) & (self._fillmin_cols | self._fillmean_cols):
            raise ValueError('Column fillna mode already defined')
        else:
            self._fillzero_cols = cols

    @property
    def missing_col_threshold(self):
        return self._missing_col_threshold

    @missing_col_threshold.setter
    def missing_col_threshold(self, thresh):
        self._missing_col_threshold = thresh

    @property
    def filter_cols(self):
        return self._filter_cols

    @filter_cols.setter
    def filter_cols(self, filter):
        self._filter_cols = filter

    def add_filter(self, filter):
        self._filter_cols.append(filter)

    @property
    def clip_cols(self):
        return self._clip_cols

    @clip_cols.setter
    def clip_cols(self, cols):
        self._clip_cols = cols

    def add_clip_col(self, var, min, max):
        self._clip_cols.append([var, min, max])

    @property
    def bin_cols(self):
        return self._bin_cols

    @bin_cols.setter
    def bin_cols(self, cols):
        self._bin_cols = cols

    def add_bin_col(self, var, bins):
        self._bin_cols.append([var, bins])

    def bin(self, abt):
        abt = abt.copy()

        for bin in self._bin_cols:
            bins = pd.cut(abt[bin[0]], bin[1])
            dummy_na = bins.isna().sum() > self._missing_col_threshold * 0.1
            bins = pd.DataFrame(pd.get_dummies(bins, dummy_na=dummy_na))
            bins.columns = [bin[0] + '_' + str(x) for x in bins.columns]
            abt = pd.concat([abt, bins], axis=1, sort=True)

        abt.drop(columns=[x[0] for x in self._bin_cols], inplace=True)

        return abt

    def clip(self, abt):
        abt = abt.copy()

        for cl in self._clip_cols:
            abt[cl[0]] = abt[cl[0]].clip(cl[1], cl[2])

        return abt

    def fillna(self, abt, train=False):
        abt = abt.copy()

        if train:
            to_fill_cols = self._fillmin_cols | self._fillmean_cols | self._fillzero_cols
            to_fill_cols = abt[to_fill_cols].isna().sum()
            to_fill_cols = to_fill_cols[to_fill_cols >= self._missing_col_threshold]
            self._missing_col_features = set(to_fill_cols.index)

        for col in self._missing_col_features:
            abt[col+'_missing'] = abt[col].isna()

        for col in self._fillmin_cols:
            abt[col] = abt[col].fillna(abt[col].min())

        for col in self._fillmean_cols:
            abt[col] = abt[col].fillna(abt[col].mean())

        for col in self._fillzero_cols:
            abt[col] = abt[col].fillna(0)

        return abt


    def onehot_encode(self, abt):
        abt = abt.copy()

        for col in self._onehot_cols:
            dummy_na = abt[col].isna().sum() > self._missing_col_threshold * 0.1
            res = pd.get_dummies(abt[col], dummy_na=dummy_na)
            res.columns = [col+'_'+str(x) for x in res.columns]
            abt = pd.concat([abt, res], axis=1, sort=True)

        abt.drop(columns=self._onehot_cols, inplace=True)

        return abt

    def filter(self, abt, train):
        abt = abt.copy()

        if train:
            filters = self._filter_cols.copy()
        else:
            filters = [x for x in self._filter_cols if x['apply'] == 'both']

        return abt


    def clean(self, abt, train):

        abt = abt.copy()

        abt.drop(columns=self._drop_cols, inplace=True)
        abt = self.clip(abt)
        abt = self.bin(abt)
        abt = self.onehot_encode(abt)
        abt = self.fillna(abt, train)
        abt = self.filter(abt, train)

        return abt


def drop_correlated_features(data, features, threshold):
    """Automatized way to drop correlated features.

    It keeps dropping features until no correlation above the threshold is left.

    Args:
        data (pd.DataFrame): Table to filter.
        features (list): Features to consider when dropping.
        threshold (float): Maximum correlation to keep.

    Returns:
        pd.DataFrame: Filtered input dataset.

    """

    # columns which shouldn't be considered
    keep_cols = [x for x in data.columns if x not in features]

    # calculate initial correlations
    correl = data[features].corr().stack().reset_index()
    correl = correl[(correl.level_0 > correl.level_1) & (correl[0].abs() > threshold)]

    # keep dropping until no correlated feature left
    features_to_drop = []
    while len(correl) > 0:
        feat_to_drop = correl.level_0.values[0]
        correl = correl[(correl.level_0 != feat_to_drop) & (correl.level_1 != feat_to_drop)]
        features_to_drop.append(feat_to_drop)

    return data[keep_cols + [x for x in features if x not in features_to_drop]].copy()
