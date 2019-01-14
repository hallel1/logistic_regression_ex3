from numpy.distutils.system_info import xft_info
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv_handle as csv_org
import lgReg_handle as lgr
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

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix= lgr.divDataByKFold(XMatrix,y,k_parameter=10)  # Define the split - into 10 folds
    #v_theta = np.zeros(df.shape[1])
    #lgr.lgReg_iter(X_train_matrix,y_train_matrix,X_test_matrix,index=0,lamda=10)
    #lgr.lgReg(X_train_matrix, y_train_matrix, X_test_matrix, k_parameter=10, lamda=0)
    lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix, k_parameter=10)