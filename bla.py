import numpy as np
from numpy import r_,c_
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn  import datasets
import csv
import csv_handle as csv_org

def sigmoid(z):
    return 1.0/(1.0+math.e**(-1*z))

#-----------------------------
def gradient(m,lambd,theta,x_train,y_train):
    theta_zeroed=theta
    theta_zeroed[0:]=0
    h=sigmoid(np.dot(x_train,theta))
    grads=(1.0/m)*np.dot(x_train.T,(h-y_train))+(float(lambd)/m)*theta_zeroed
    return grads

##--------------------MAIN------------------------
if __name__ == "__main__":
    lambd=0.2
    alpha=0.1
    iterations=1000
    # read data set
    path = 'hearts.csv'
    df_org = pd.read_csv(path)
    dfCopy = df_org.__deepcopy__()
    df = dfCopy.replace(np.nan, '', regex=True)  # replace nan values with ''
    csv_org.insert_col_df(df)
    csv_org.normalizationAll(df)
    XMatrix = csv_org.x_matrix(df)
    #definr theta
    theta = np.random.rand(len(XMatrix[0]))
    # print(theta)

    #split dataset

    y = csv_org.y_vector(df)
    X_train, X_test, Y_train, y_test = train_test_split(XMatrix, y, test_size=0.30, random_state=0)

    cost = []
#cange from list to arrays
    x_train=np.copy(X_train)
    # print(x_train.shape)
    y_train=np.copy( Y_train)


    #Bath gradient descent

    for i in range(iterations):
        grad=gradient(x_train.shape[0],lambd,theta,x_train,y_train )
        #UPDATE theta
        theta[0:]=theta[0:]-alpha*grad[0:]
        theta[1:]=theta[1:]-alpha*grad[1:]
    p=np.dot(X_test,theta)
    p[p>=0.5]=1
    p[p<0.5]=0
    # clf= linear_model.LogisticRegression(penalty='l1',fit_intercept=True,n_jobs=-1)
    # clf.fit(x_train,y_train)
    # clf_prediction=[]
    # print(y_train.shape[0])
    # for i in range(y_train.shape[0]):
    #     clf_prediction.append(clf.predict(x_train[i:].reshape(1,-1)))





