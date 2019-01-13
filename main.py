from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import csv_handle as csv_org

path = 'hearts.csv'
df_org = pd.read_csv(path)
dfCopy=df_org.__deepcopy__()

df = dfCopy.replace(np.nan, '', regex=True)# replace nan values with ''
print(df)