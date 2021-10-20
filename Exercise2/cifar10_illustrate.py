import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random, randrange
import operator


# TASK 2
# returns classification accuracy(%) of provided labels
def class_acc(pred, qt):
    # check for empty or different sized lists, return -1 if error found
    if (len(pred) == 0 or len(qt) == 0):
        return -1
    elif (len(pred) != len(qt)):
        return -1

    correct = 0
    wrong = 0
    for i in range(0, len(pred)):
        if pred[i] == qt[i]:
            correct += 1
        else:
            wrong += 1

    accuracy = correct / (correct + wrong) * 100
    return accuracy


def cifar10_classifier_random(x):
    possibleLabels = 10
    return randrange(0, possibleLabels)


def cifar10_classifier_1nn(x, trdata, trlabels):
    trainingSamples = trdata[0:50000]
    trainingSampleLabels = trlabels[0:50000]

    # go through training data and compare all to x(test image)
    smallestDistance = 1000000000000
    x = x.astype('int')
    for index, sample in enumerate(trainingSamples):
        sample = sample.astype('int')
        temp = np.subtract(x, sample)
        temp = np.power(temp, 2)
        totalEuclideanDistance = np.sum(temp)

        if (totalEuclideanDistance < smallestDistance):
            smallestDistance = totalEuclideanDistance
            bestMatch = index

    return trainingSampleLabels[bestMatch]


# finds A (i.e. 25) nearest neighbours and scores/points them. nearer = more points
# after points, the most points collected label "wins".
# This was a test and ended up being slower and worse than 1nn
def cifar10_classifier_Ann(x, trdata, trlabels):
    indexDistancePairs = []
    trainingSamples = trdata[0:50000]
    trainingSampleLabels = trlabels[0:50000]

    # go through training data and compare all to x(test image)
    smallestDistance = 1000000000000
    x = x.astype('int')
    for index, sample in enumerate(trainingSamples):
        sample = sample.astype('int')

        temp = np.subtract(x, sample)
        temp = np.power(temp, 2)
        totalEuclideanDistance = np.sum(temp)

        indexDistancePairs.append((index, totalEuclideanDistance))

    indexDistancePairs.sort(key=operator.itemgetter(1))

    labelPoints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    points = 25
    for idx, val in indexDistancePairs:
        labelidx = trainingSampleLabels[idx]
        labelPoints[labelidx] += (points ** 2)
        points -= 1

        if (points == 0):
            break

    # return label with most points
    maxIdx = np.argmax(labelPoints)
    return maxIdx


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# TASK 2 SCRIPT
datadict = unpickle('cifar-10-batches-py/test_batch')
X = datadict["data"]
Y = datadict["labels"]
print(f"Task 2 class_acc accuracy: {class_acc(Y, Y)}%")

# TASK 3 SCRIPT
# inputs all CIFAR-10 test samples and evaluates classification accuracy
testSamples = unpickle('cifar-10-batches-py/test_batch')["data"]
gtLabels = unpickle('cifar-10-batches-py/test_batch')["labels"]
randomLabels = []
for sample in testSamples:
    randomLabels.append(cifar10_classifier_random(sample))
print(f"Task 3 Random label accuracy: {class_acc(randomLabels, gtLabels)}%")

# combine all training data to one ndarray
trainingData1 = unpickle('cifar-10-batches-py/data_batch_1')["data"]
trainingData2 = unpickle('cifar-10-batches-py/data_batch_2')["data"]
trainingData3 = unpickle('cifar-10-batches-py/data_batch_3')["data"]
trainingData4 = unpickle('cifar-10-batches-py/data_batch_4')["data"]
trainingData5 = unpickle('cifar-10-batches-py/data_batch_5')["data"]
trainingData = np.concatenate((trainingData1, trainingData2, trainingData3, trainingData4, trainingData5))

# combine all training data labels to one ndarray
trainingLabels1 = unpickle('cifar-10-batches-py/data_batch_1')["labels"]
trainingLabels2 = unpickle('cifar-10-batches-py/data_batch_2')["labels"]
trainingLabels3 = unpickle('cifar-10-batches-py/data_batch_3')["labels"]
trainingLabels4 = unpickle('cifar-10-batches-py/data_batch_4')["labels"]
trainingLabels5 = unpickle('cifar-10-batches-py/data_batch_5')["labels"]
trainingLabels = np.concatenate((trainingLabels1, trainingLabels2, trainingLabels3, trainingLabels4, trainingLabels5))

# testing data
trainingDataforTesting = unpickle('cifar-10-batches-py/test_batch')

# Task 4 SCRIPT
usedTestData = 2  # give number between 1-10000

predLabels = []
for testSample in trainingDataforTesting["data"][0:usedTestData]:
    predictedLabel = cifar10_classifier_1nn(testSample, trainingData, trainingLabels)
    predLabels.append(predictedLabel)
accuracy = class_acc(predLabels, trainingDataforTesting["labels"][0:usedTestData])
print(f"Task 4 1nn accuracy: {accuracy}% with 50 000 training data and {usedTestData} testing data")
