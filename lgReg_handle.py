from sklearn.model_selection import KFold

def kFold(XMatrix,y):
    kf = KFold(n_splits=10)  # Define the split - into 10 folds
    kf.get_n_splits(XMatrix)  # returns the number of splitting iterations in the cross-validator
    print(kf)

    for train_index, test_index in kf.split(XMatrix):
        # print('TRAIN:', train_index, 'TEST:', test_index)
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


