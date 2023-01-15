import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split


def knn(train, temp_p, ord, flag):
    """
        This function calculates the distance between a given point (temp_p) and all the points in the training set (train).
        The distance metric used is the L2-norm (Euclidean distance) and it's determined by the variable 'ord' passed to the function.
        The 'flag' variable is used to determine whether the point has 2 or 3 dimensions. The function returns a list of tuples,
        each containing the distance and label of a point in the training set.
    """
    distances = []
    if flag:
        pointA = np.array([temp_p.x, temp_p.y, temp_p.z])
    else:
        pointA = np.array([temp_p.x, temp_p.y])
    for point in train.itertuples():
        if flag:
            pointB = np.array([point.x, point.y, point.z])
        else:
            pointB = np.array([point.x, point.y])
        D = np.linalg.norm(pointA - pointB, ord=ord)
        distances.append((D, point.label))
    distances.sort(key=lambda tup: tup[0])
    return distances


def getPred(train, test, index_odd, index_p, flag):
    """
        This function takes in the training set, test set, number of nearest neighbors (index_odd),
        the distance metric (index_p) and the flag variable. It returns an array of predictions for each point in the test set,
        based on the majority class of its k nearest neighbors in the training set.

    """
    pred = np.zeros(test.shape[0])

    index_ = 0
    minus_idx = 0
    plus_idx = 1
    # for p in test:
    for point in train.itertuples():
        dis = knn(train, point, index_p, flag)
        minus_plus = [0, 0]
        for i in range(index_odd):
            if dis[i][1] == -1:
                minus_plus[minus_idx] += 1
            else:
                minus_plus[plus_idx] += 1

        if minus_plus[minus_idx] > minus_plus[plus_idx]:
            pred[index_] = -1
        else:
            pred[index_] = 1
        index_ += 1
    return pred


def rountVal(val):
    """
        This function return value round to 4 numbers of digits

    """
    return '{:.4f}'.format(val)


def run(_points, times, maxK, flag):
    """
        This function runs the k-NN algorithm multiple times (determined by the 'times' variable) and calculates the train and test 
        errors for different values of k (1,3,5,7,9) and different distance metrics (p = 1,2,inf). 
        It also prints the train and test errors in a tabular format

    """
    final_train_error = np.zeros((5, 3))
    final_test_error = final_train_error.copy()
    length = int(len(_points) / 2)
    for i in range(times):
        x_train, x_test, y_train, y_test = train_test_split(_points.drop(columns=['label']),
                                                            _points['label'],
                                                            test_size=0.5, stratify=_points['label'])
        train = pd.concat([x_train, y_train], axis=1)
        test = pd.concat([x_test, y_test], axis=1)
        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])
        P = [1, 2, np.inf]
        K = range(1, maxK + 1, 2)
        trainErrors = np.zeros((5, 3))
        testErrors = trainErrors.copy()
        index_odd = 0
        for k in K:
            testE = np.zeros(3)
            trainE = testE.copy()
            index_p = 0
            for p in P:
                train_pred = getPred(train, train, k, p, flag)
                test_pred = getPred(train, test, k, p, flag)
                idx = 0
                for j in range(length):
                    if test.loc[j, 'label'] != test_pred[idx]:
                        testE[index_p] += 1
                    if train.loc[j, 'label'] != train_pred[idx]:
                        trainE[index_p] += 1
                    idx += 1
                index_p += 1
            trainErrors[index_odd] = trainE / length
            testErrors[index_odd] = testE / length
            index_odd += 1
        final_test_error += testErrors
        final_train_error += trainErrors

    final_test_error = final_test_error / times
    final_train_error = final_train_error / times

    print("\n\nResult for true error:\nk\t\tp=1\t\tp=2\t\tp=inf")
    for i in range(times):
        print(str(rountVal(K[i])) + "\t\t" + str(rountVal(final_train_error[i][0])) + "\t\t" +
              str(rountVal(final_train_error[i][1])) + "\t\t" + str(rountVal(final_train_error[i][2])))
    print("")
    print("\n\nResult for empirical error:\nk\t\tp=1\t\tp=2\t\tp=inf")
    for i in range(times):
        print(str(rountVal(K[i])) + "\t\t" + str(rountVal(final_test_error[i][0])) + "\t\t" +
              str(rountVal(final_test_error[i][1])) + "\t\t" + str(rountVal(final_test_error[i][2])))
    print("Each row represents the k value (1,3,5,7,9) and each column represents the p value (1,2,inf)")

    print("\n\nThe difference between empirical and true error:\nk\t\tp=1\t\tp=2\t\tp=inf")
    diff = np.absolute(final_train_error - final_test_error)
    for i in range(times):
        print(str(rountVal(K[i])) + "\t\t" + str(rountVal(diff[i][0])) + "\t\t" +
              str(rountVal(diff[i][1])) + "\t\t" + str(rountVal(diff[i][2])))


if __name__ == '__main__':
    haberman = str('haberman.data')
    squares = str('squares.txt')
    flag = False
    if flag:
        points = pd.read_csv(haberman, sep=",", header=None,
                             names=["x", "y", "z", "label"])
        points.loc[points["label"] == 2, "label"] = -1
    else:
        points = pd.read_csv(squares, sep=" ", header=None,
                             names=["x", "y", "label"])
        points.loc[points["label"] == 0, "label"] = -1
    run(points, 5, 9, flag)
