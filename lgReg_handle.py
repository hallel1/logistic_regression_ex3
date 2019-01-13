from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def kFold(XMatrix,y,k_parameter):#div data to test and train by k fold

    kf = KFold(n_splits=k_parameter)  # Define the split - into 10 folds
    kf.get_n_splits(XMatrix)  # returns the number of splitting iterations in the cross-validator
    # print(kf)
    X_train_matrix = []
    X_test_matrix = []
    y_train_matrix = []
    y_test_matrix = []

    for train_index, test_index in kf.split(XMatrix):
        #Each iteration get new: X_train,y_train,X_test,y_test

        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(train_index.__len__()):
            indexTrain = train_index[i]
            X_train.append(XMatrix[indexTrain])
            y_train.append(y[indexTrain])

        for j in range(test_index.__len__()):
            indexTest = test_index[j]
            X_test.append(XMatrix[indexTest])
            y_test.append(y[indexTest])

        X_train_matrix.append(X_train)
        X_test_matrix.append(X_test)
        y_train_matrix.append(y_train)
        y_test_matrix.append(y_test)

        return (X_train_matrix,X_test_matrix,y_train_matrix,y_test_matrix)
#------------------------------------------------------------------------

def lgReg_iter(X_train_matrix,y_test_matrix,index,lamda):
    c_parameter = 1 / lamda
    logreg = LogisticRegression(C=c_parameter, solver='lbfgs').fit(X_train_matrix[index], y_test_matrix[index])