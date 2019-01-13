from numpy.distutils.system_info import xft_info
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

    csv_org.insert_col_df(df)
    csv_org.normalizationAll(df)

    print(df)

    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)
    '''
    X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size=0.30, random_state=42)
     
    v_theta = np.zeros(df.shape[1])  # logreg.random_theta(df)


    lamda=10
    c_parameter=1/lamda
    logreg = LogisticRegression(C=c_parameter ,solver='lbfgs').fit(X_train, y_train)
    print(logreg)

    print(X_test)
    print(logreg.predict_proba(X_test))#check the result of the model about x-test
    '''
    kf = KFold(n_splits=10)  # Define the split - into 10 folds
    kf.get_n_splits(XMatrix)  # returns the number of splitting iterations in the cross-validator
    print(kf)


    for train_index, test_index in kf.split(XMatrix):
         #print('TRAIN:', train_index, 'TEST:', test_index)
         X_train = []
         X_test = []
         y_train = []
         y_test = []
         for i in range(train_index.__len__()):
              indexTrain = train_index[i]
              X_train.append(XMatrix[indexTrain])
              y_train.append(XMatrix[indexTrain])

         for j in range(test_index.__len__()):
              indexTest=test_index[j]
              X_test.append(XMatrix[indexTest])
              y_test.append(XMatrix[indexTest])




         #X_train = XMatrix[train_index]
        # X_test = XMatrix[test_index]
        # y_train = y[train_index]
        # y_test = y[test_index]

