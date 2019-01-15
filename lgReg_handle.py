from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

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
def average(vec):#average of vector
    sum = 0
    length=len(vec)
    for i in range(length):
        sum += vec[i]
    average = sum / length
    return average

def indexMinElement(vec):#return the index of min element in vector
    minVal = vec[0]
    length=len(vec)
    indexMin=0
    for i in range(1,length):
        if minVal> vec[i]:
            minVal=vec[i]
            indexMin=i
    print('min',minVal,'i',indexMin)
    return indexMin
#-------------------------------------
def k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10):
    #C_param_range = [np.inf,1000,500,200,100,10,1,0.1,0.01,0.001,0.0001]
    C_param_range = [np.inf,1000,100,10,1,0.1,0.01,0.001,0.0001]
    avg=[]
    testErrOneModel = [0.0] * k_parameter
    testErrAllModels = []
    #trainErr= [0.0] * k_parameter
    for c in C_param_range:
        for i in range(k_parameter):
            print('i ',i,' c ',c)

            logreg = LogisticRegression(C=c, solver='lbfgs', penalty='l2').fit(X_train_matrix[i],y_train_matrix[i])
            #errI=logreg.predict_proba(X_test_matrix[i])
            errI=logreg.predict(X_test_matrix[i])
            predict_train=logreg.predict(X_train_matrix[i])
            #err.append(logreg.predict_proba(X_test_matrix[i]))
            print('err ',errI)
            print("yts",y_test_matrix[i])
            testErrOneModel[i]= float(sum(errI != y_test_matrix[i])) / len(y_test_matrix[i])
        print('testErrOneModel', testErrOneModel)
        testErrAllModels.append(np.mean(testErrOneModel))
    print('testErrAllModels',testErrAllModels)
    indexBetterModel = indexMinElement(testErrAllModels)

    optimalLamda = C_param_range[indexBetterModel]
    print('The optimal lamda', optimalLamda)
    print('The average error of the model with this lamda is:', testErrAllModels[indexBetterModel])
    print('The average error of the model with lamda=0 is:', testErrAllModels[0])



        #trainErr[i] = float(sum(predict_train != y_train_matrix[i])) / len(y_train_matrix[i])
        # print("sum error", float(sum(errI != y_test_matrix[i])),'len ',len(y_test_matrix[i]))
    #     print("test Err",i,"=", testErr[i])
    # print(testErr)
    # #print(trainErr)
    # indexBetterModel=indexMinElement(testErr)
    # optimalLamda=C_param_range[indexBetterModel]
    # print('optimal',optimalLamda)
    # print('min', testErr[indexBetterModel],'i',indexBetterModel)

    # print('The optimal lamda',optimalLamda)
    # print('The average error of the model with this lamda is',testErr[indexBetterModel])
    #errOptimalLamda = logreg.predict_proba(testErr[indexBetterModel])

    #
    # print("summary:")
    # print("average train err =", np.mean(trainErr) * 100, "%")
    # print("average test err =", np.mean(testErr) * 100, "%")

    #draw_graph(testErr,trainErr, C_param_range)

        #averageErr=average(err)
        #avg.append(averageErr)
    #print('avg ',avg)
    #we want the c that give min avg
    #for i in range(len(avg)):

# ------------------------------------------------------------------------
def draw_graph(v_testErr,v_trainErr,v_C):
    plt.title(" error for given lambdas" )
    plt.plot(v_C,v_testErr,label='test error')
    plt.plot(v_C, v_trainErr, label='train error')
    plt.xlabel('lambdas')
    plt.ylabel('erors')
    # plt.text()

    plt.legend()
    plt.show()



# def lgReg_iter(X_train_matrix,y_train_matrix,index,c_parameter):
   #  logreg = LogisticRegression(C=c_parameter, solver='lbfgs',penalty ='L2').fit(X_train_matrix[index], y_train_matrix[index])
# ------------------------------------------------------------------------


#------------------------------------------------------------------------
#def optimalLamba(X_train_matrix, y_train_matrix, k_parameter, lamdaInitial):
