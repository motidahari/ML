from sklearn.model_selection import *
from typing import List, Tuple
from numpy.linalg import lstsq
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools


INF = float("inf")


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Point(x={self.x}, y={self.y})"


class Line:
    def __init__(self, p1: Point, p2: Point, orient: bool = True):
        self.orient = orient
        self.parallel = False
        if p1.y == p2.y:
            self.weight = 0
            self.bias = p1.y
        elif p1.x == p2.x:
            self.parallel = True
            self.bias = p1.x
        else:
            self.weight, self.bias = leastSquaresRegression(
                p1.x, p2.x, p1.y, p2.y)

    def getTag(self, p: Point) -> int:
        if self.parallel:
            return 1 if (p.x < self.bias) == self.orient else -1
        above = isAbove(p.x, p.y, self.weight, self.bias)
        return 1 if (above == self.orient) else -1


def leastSquaresRegression(x1, x2, y1, y2):
    x = np.array([x1, x2])
    y = np.array([y1, y2])
    xV = np.vstack([x, np.ones(len(x))]).T
    weight, bias = np.linalg.lstsq(xV, y, rcond=None)[0]
    return weight, bias


def isAbove(px, py, weight, bias):
    return py - (px * weight + bias) >= 0


def createData():
    points = []
    tags = []
    with open("squares.txt") as fp:
        for line in fp:
            x, y, tag = map(float, line.split())
            points.append(Point(x, y))
            tags.append(-1 if tag != 1 else 1)
    return np.array(points), np.array(tags)


def generatelines(points) -> list:

    lines = []
    for p1, p2 in itertools.combinations(points, 2):
        lines += [Line(p1, p2, True), Line(p1, p2, False)]
    return lines


def calculateClassificationError(
    hypos: List[Point], points: List[Point], tags: List[int], alphas: List[float]
) -> float:
    n = len(points)
    err = 0
    for i in range(n):
        if computeDecision(hypos, alphas, points[i]) != tags[i]:
            err += 1
    return err / n


def computeDecision(hypos, alphas, point) -> float:
    sigma = 0
    for i in range(len(hypos)):
        computeDecision_for_h = hypos[i].getTag(point)
        sigma = sigma + computeDecision_for_h * alphas[i]
    if sigma > 0:
        return 1
    if sigma < 0:
        return -1
    return 1


def AddErrors(avgs, current):
    for i in range(8):
        avgs[i] += current[i]


def adaboost(points: List[Point], labels: List[int], mComplexity: int) -> Tuple[List[float], List[float]]:
    trainPoints, test_points, trainLabels, testLabels = train_test_split(
        points, labels, test_size=0.5, shuffle=True)
    hypos = generatelines(trainPoints)
    n = len(trainPoints)
    weights = [1 / n] * n
    alphas, good_hyps, errorsTrain, errorTestors = [], [], [], []
    for i in range(mComplexity):
        minErr = INF
        for h in hypos:
            err = sum(
                weights[j] * (h.getTag(trainPoints[j]) != trainLabels[j])
                for j in range(n)
            )
            if minErr > err:
                minErr = err
                classifier = h
        good_hyps.append(classifier)
        alpha = (1 / 2) * math.log((1 - minErr) / minErr)
        alphas.append(alpha)

        for m in range(n):
            weights[m] = weights[m] * math.exp(
                -alpha * classifier.getTag(trainPoints[m]) * trainLabels[m]
            )
        weights = [weight / sum(weights) for weight in weights]
        trainErrorCalculation = calculateClassificationError(
            good_hyps, trainPoints, trainLabels, alphas)
        calc_errorTest = calculateClassificationError(
            good_hyps, test_points, testLabels, alphas)
        errorsTrain.append(trainErrorCalculation)
        errorTestors.append(calc_errorTest)

    return errorsTrain, errorTestors


if __name__ == '__main__':
    points, lables = createData()
    trainAvgs = [0] * 8
    testAvgs = [0] * 8
    iterations = 50
    for i in range(iterations):
        trainPoints, test_points, train_lables, test_lables = train_test_split(points, lables, test_size=0.5,
                                                                               shuffle=True)
        errorTrain, errorTest = adaboost(trainPoints, train_lables, 8)
        AddErrors(trainAvgs, errorTrain)
        AddErrors(testAvgs, errorTest)
    trainAvgs = [avg / iterations for avg in trainAvgs]
    testAvgs = [avg / iterations for avg in testAvgs]
    print('testAvgs:', testAvgs)
    print('trainAvgs:', trainAvgs)

    numHyps = []
    for i in range(1, 9):
        numHyps.append(i)
    plt.xlabel('Number Of Hypos')
    plt.ylabel('Error')
    plt.plot(numHyps, errorTest, color='green',
             linewidth=5, label='test errors')
    plt.plot(numHyps, errorTrain, color='black',
             linewidth=5, label='train errors')
    plt.legend()
    plt.show()
