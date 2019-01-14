from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def divDataByKFold(XMatrix,y,k_parameter):#div data to test and train by k fold

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
            print('')

        for j in range(test_index.__len__()):
            indexTest = test_index[j]
            X_test.append(XMatrix[indexTest])
            y_test.append(y[indexTest])

        X_train_matrix.append(X_train)
        X_test_matrix.append(X_test)
        y_train_matrix.append(y_train)
        y_test_matrix.append(y_test)

    # print(X_train_matrix)
    # print('############')
    # print(X_test_matrix)
    # print('%%%%%%%%')
    # print(y_train_matrix)
    # print('^^^^^^^^')
    # print(y_test_matrix)


    return (X_train_matrix,X_test_matrix,y_train_matrix,y_test_matrix)
#------------------------------------------------------------------------
def k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix, k_parameter=10):
    C_param_range = [1000,100,10,1,0.1,0.01]
    avg=[]
    for c in C_param_range:
        err=[]
        for i in range(k_parameter):
            #print('i ',i,' c ',c)

            logreg = LogisticRegression(C=c, solver='lbfgs', penalty='l2').fit(X_train_matrix[i],y_train_matrix[i])
            err.append(logreg.predict_proba(X_test_matrix[i]))
        print('err ',err)
        print('sum(err) ', sum(err),' len(err) ', len(err))
        avg.append(sum(err)/len(err))
    print('avg ',avg)
    #we want the c that give min avg
    #for i in range(len(avg)):


# def lgReg_iter(X_train_matrix,y_train_matrix,index,c_parameter):
   #  logreg = LogisticRegression(C=c_parameter, solver='lbfgs',penalty ='L2').fit(X_train_matrix[index], y_train_matrix[index])
# ------------------------------------------------------------------------


#------------------------------------------------------------------------
#def optimalLamba(X_train_matrix, y_train_matrix, k_parameter, lamdaInitial):
