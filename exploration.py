import pandas as pd
from pandas_profiling import ProfileReport
from tqdm import tqdm


train = pd.read_csv('data/train.csv')
train.shape

test = pd.read_csv('data/test.csv')
test.shape

# profile = ProfileReport(train.head(1000), minimal=True)
# profile.to_file('EDA/raw_report.html')

# train.isna().sum().sort_values(ascending=False)

train.article_id.nunique() == len(train)

corr = train.corr().stack().reset_index()
corr[corr.level_0 > corr.level_1].sort_values(0, ascending=False)
