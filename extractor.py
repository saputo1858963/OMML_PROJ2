"""
@author: Ilaria

This is the code you need to run to import the dataset.
Make sure the folder 'FashionMNIST' is inside your project directory (i.e. at the same level of your Q_i folders).

You can copy and paste the extract_data() function in your function files.

If you keep the FashionMNIST folder inside your project directory at the same level as your Q_i folders
containing the extract_data function, the code should run without needing any modification.

Otherwise, you may need to adjust lines 27 and 28 to match your local path,
but make sure that the code you submit can be executed on any system without needing manual path modifications.
"""

import numpy as np
import pandas as pd
import os


def extract_data():
    """
    Load and prepare FashionMNIST folder containing the training and testing data (csv files).
    Make sure the folder 'FashionMNIST' is inside your project directory (i.e. at the same level of your Q_i folders).
    """

    train = pd.read_csv(os.path.join('..', 'FashionMNIST', 'fashion-mnist_train.csv'))
    test = pd.read_csv(os.path.join('..', 'FashionMNIST', 'fashion-mnist_test.csv'))

    # We need to extract only the items with label 2 and 3:
    indexes = [2, 3]

    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:, 1:]
    Y = train_set.values[:, 0]

    X_Test = test_set.values[:, 1:]
    Y_Test = test_set.values[:, 0]

    ind_2 = np.where(Y == 2)
    ind_2_Test = np.where(Y_Test == 2)

    ind_3 = np.where(Y == 3)
    ind_3_Test = np.where(Y_Test == 3)

    """
    Only a subset of 1000 samples per class will be used.
    To train a SVM in case of binary classification we have to convert the labels of the two classes of interest into '+1' and '-1'.
    """

    X2 = X[ind_2[0][:1000]]
    Y2 = np.ones(1000)

    X3 = X[ind_3[0][:1000]]
    Y3 = -np.ones(1000)

    X2test = X_Test[ind_2_Test[0][:200]]
    Y2test = np.ones(X2test.shape[0])

    X3test = X_Test[ind_3_Test[0][:200]]
    Y3test = -np.ones(X3test.shape[0])

    X_train = np.concatenate((X2, X3))
    Y_train = np.concatenate((Y2, Y3))

    X_test = np.concatenate((X2test, X3test))
    Y_test = np.concatenate((Y2test, Y3test))

    # Reshuffle training data
    np.random.seed(1)
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    return X_train, Y_train, X_test, Y_test

