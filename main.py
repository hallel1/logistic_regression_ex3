from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import csv_handle as csv_org
import numpy as np
import pandas as pd
from pandas import DataFrame
##--------------------MAIN------------------------
path = 'hearts.csv'
df_org = pd.read_csv(path)
dfCopy=df_org.__deepcopy__()
if __name__ == "__main__":
    df = dfCopy.replace(np.nan, '', regex=True)# replace nan values with ''

    csv_org.insert_col_df(df2)
    csv_org.normalizationAll(df2)
    XMatrix = csv_org.x_matrix(df2)
    y = csv_org.y_vector(df2)
    X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size=0.30, random_state=42)
    v_theta = np.zeros(df.shape[1])  # logreg.random_theta(df2)

    #clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)