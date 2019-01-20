import csv_handle as csv_org
import lgReg_handle as lgr
import numpy as np
import pandas as pd

##--------------------MAIN------------------------
path = 'hearts.csv'
df_org = pd.read_csv(path)
dfCopy=df_org.__deepcopy__()
if __name__ == "__main__":
    df = dfCopy.replace(np.nan, '', regex=True)# replace nan values with ''

    csv_org.insert_col_df(df)
    csv_org.normalizationAll(df)# normalization data


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)


    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix\
        = lgr.divDataByKFold(XMatrix,y,k_parameter=10)  # Define the split - into 10 folds

    C_param_range, testErrAllModels,optimalLambda=\
        lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10)


    lgr.draw_graph(C_param_range, testErrAllModels)
    lgr.raph_learning_groups(XMatrix,y,optimalLambda)

