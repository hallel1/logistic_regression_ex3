from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


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

# def lgReg_iter(X_train_matrix,y_train_matrix,index,c_parameter):
#     logreg = LogisticRegression(C=c_parameter, solver='lbfgs').fit(X_train_matrix[index], y_train_matrix[index])
# ------------------------------------------------------------------------

def lgReg(X_train_matrix, y_train_matrix,X_test_matrix ,k_parameter, lamda):
    '''
    c_parameter = 1 / lamda
    for i in range(k_parameter):
        #lgReg_iter(X_train_matrix, y_train_matrix, index=i, c_parameter=c_parameter)
        logreg = LogisticRegression(C=c_parameter, solver='lbfgs', penalty='l2')#the penalty parameter - for the norm 2
        logreg.fit(X_train_matrix[i], y_train_matrix[i])
    '''
    lamda=0.0001
    c_parameter = 1 / lamda


   # sepal_acc_table = pd.DataFrame(columns=['C_parameter', 'Accuracy'])
   # sepal_acc_table['C_parameter'] = C_param_range
    plt.figure(figsize=(10, 10))

    indexTrain=0
    j = 0
    while c_parameter>100: #change
        # Apply logistic regression model to training data
        logreg = LogisticRegression(penalty='l2', C=c_parameter, random_state=0,solver='lbfgs')
        logreg.fit(X_train_matrix[indexTrain], y_train_matrix[indexTrain])

        # Predict using model
        y_pred_sepal = logreg.predict(X_test_matrix[indexTrain])

        # Saving accuracy score in table
        #sepal_acc_table.iloc[j, 1] = accuracy_score(y_test_sepal, y_pred_sepal)
        j += 1

        # Printing decision regions
        plt.subplot(3, 2, j)
        plt.subplots_adjust(hspace=0.4)

        '''plot_decision_regions(X=X_combined_sepal_standard
                              , y=Y_combined_sepal
                              , classifier=logreg
                              , test_idx=range(105, 150))'''
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        #plt.title('C = %s' % i)

        lamda = lamda*10
        c_parameter = 1 / lamda
#------------------------------------------------------------------------
#def optimalLamba(X_train_matrix, y_train_matrix, k_parameter, lamdaInitial):
