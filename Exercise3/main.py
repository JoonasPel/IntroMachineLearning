import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from skimage import transform
from PIL import Image
from collections import Counter
import scipy.stats as sc
import math


# rescales images from 32x32 to 1x1
def cifar10_color(X):
    print(f"resizing {len(X)} images")

    resizedImages = []
    for image in X:
        splitImage = np.split(image, 3)
        meanR = np.mean(splitImage[0]).astype('uint8')
        meanG = np.mean(splitImage[1]).astype('uint8')
        meanB = np.mean(splitImage[2]).astype('uint8')
        resizedImages.append([meanR, meanG, meanB])
    # convert array to ndarray
    resizedImages = np.asarray(resizedImages)

    print("resizing done\n")
    return resizedImages


def cifar_10_bayes_learn(Xf, Y):
    print("learning in cifar_10_bayes_learn")

    # Dict where labels are keys and values are their RGB values
    # e.g. { 0 : [[Red values], [Green values], [Blue values]] }
    #      { 1 : [[Red values], [Green values], [Blue values]] }
    meanValues = {}
    for i in range(0, 10):
        meanValues[i] = [[], [], []]
    # add colours to correct labels
    for index, image in enumerate(Xf):
        label = Y[index]
        meanValues[label][0].append(image[0])
        meanValues[label][1].append(image[1])
        meanValues[label][2].append(image[2])

    # calculate covariances
    covariances = []
    for index in range(0, 10):
        valuesRGB = meanValues[index]
        valuesRGB = np.asarray(valuesRGB)
        covariances.append(np.cov(valuesRGB))


    # calculate means and replace color arrays with them
    for label in range(0, 10):
        meanValues[label][0] = np.mean(meanValues[label][0])
        meanValues[label][1] = np.mean(meanValues[label][1])
        meanValues[label][2] = np.mean(meanValues[label][2])

    # priors, index = label
    labelFrequency = sorted((Counter(Y)).items())
    priors = []
    for label in labelFrequency:
        priors.append(label[1])
    priors = np.divide(priors, len(Xf))

    print("learning done\n")
    return meanValues, covariances, priors


def cifar10_classifier_bayes(x, mu, sigma, p):
    classProbabilities = []
    for i in range(0, 10):
        norm = sc.multivariate_normal.pdf(x, mu[i], sigma[i])
        classProb = norm * p[i]
        classProbabilities.append(classProb)
    return np.argmax(classProbabilities)


# Xp 1x1x3 images, Y labels
def cifar_10_naivebayes_learn(Xp, Y):
    print("learning in cifar_10_naivebayes_learn")

    # Dict where labels are keys and values are their RGB values
    # e.g. { 0 : [[Red values], [Green values], [Blue values]] }
    #      { 1 : [[Red values], [Green values], [Blue values]] }
    meanValues = {}
    for i in range(0, 10):
        meanValues[i] = [[], [], []]
    # add colours to correct labels
    for index, image in enumerate(Xp):
        label = Y[index]
        meanValues[label][0].append(image[0])
        meanValues[label][1].append(image[1])
        meanValues[label][2].append(image[2])

    # dict for variances
    varianceValues = {}
    for i in range(0, 10):
        varianceValues[i] = [[], [], []]
    # calculate variances
    for label in range(0, 10):
        varianceValues[label][0] = np.var(meanValues[label][0])
        varianceValues[label][1] = np.var(meanValues[label][1])
        varianceValues[label][2] = np.var(meanValues[label][2])

    # calculate means and replace color arrays with them
    for label in range(0, 10):
        meanValues[label][0] = np.mean(meanValues[label][0])
        meanValues[label][1] = np.mean(meanValues[label][1])
        meanValues[label][2] = np.mean(meanValues[label][2])

    # priors, index = label
    labelFrequency = sorted((Counter(Y)).items())
    priors = []
    for label in labelFrequency:
        priors.append(label[1])
    priors = np.divide(priors, len(Xp))

    print("learning done\n")
    return meanValues, varianceValues, priors


def cifar10_classifier_naivebayes(x, mu, sigma, p):
    # index = class/label
    classProbabilities = []
    for i in range(0, 10):
        normRED = sc.norm.pdf(x[0], mu[i][0], math.sqrt(sigma[i][0]))
        normGREEN = sc.norm.pdf(x[1], mu[i][1], math.sqrt(sigma[i][1]))
        normBLUE = sc.norm.pdf(x[2], mu[i][2], math.sqrt(sigma[i][2]))
        priorProb = p[i]

        classProb = normRED * normGREEN * normBLUE * priorProb
        classProbabilities.append(classProb)
    return np.argmax(classProbabilities)


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


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


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

testingData = unpickle('cifar-10-batches-py/test_batch')

# Used data sizes
usedTestImages = 10000  # 1-10000
usedTrainingImages = 50000

# resize images to 1x1 for Tasks 1&2
trainingImages1x1 = cifar10_color(trainingData[0:usedTrainingImages])
testImages1x1 = cifar10_color(testingData["data"][0:usedTestImages])

# parameters for naive bayes and bayes
means, variances, priors = cifar_10_naivebayes_learn(trainingImages1x1, trainingLabels[0:usedTrainingImages])
meansB, covariances, priorsB = cifar_10_bayes_learn(trainingImages1x1, trainingLabels[0:usedTrainingImages])

# calculate predicted labels
print("calculating labels")
predLabelsNaive = []
predLabelsBayes = []
for testImage in testImages1x1:
    predLabelsNaive.append(cifar10_classifier_naivebayes(testImage, means, variances, priors))
    predLabelsBayes.append(cifar10_classifier_bayes(testImage, meansB, covariances, priorsB))

# calculate accuracies
accuracyNaive = class_acc(predLabelsNaive, testingData["labels"][0:usedTestImages])
accuracyBayes = class_acc(predLabelsBayes, testingData["labels"][0:usedTestImages])

print(f"bayes naive accuracy with {usedTestImages} test images: {accuracyNaive}%")
print(f"bayes accuracy with {usedTestImages} test images: {accuracyBayes}%\n")



