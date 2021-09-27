import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from skimage import transform, img_as_ubyte
from PIL import Image
from collections import Counter
import scipy.stats as sc
import math

def cifar10_WxW_color(X, width):
    print(f"resizing {len(X)} images: 32x32 -> {width}x{width}")

    resizedImages = []
    for image in X:
        reshapedIMG = image.reshape(3, 32, 32).transpose(1, 2, 0)
        resizedIMG = img_as_ubyte(transform.resize(reshapedIMG, (width, width), anti_aliasing=True))
        # "extracting" colors to own arrays, better solution needed for performance
        reds, greens, blues = [], [], []
        for row in resizedIMG:
            for column in row:
                reds.append(column[0])
                greens.append(column[1])
                blues.append(column[2])
        resizedImages.append(np.concatenate((reds, greens, blues)))
    # convert array to ndarray
    resizedImages = np.asarray(resizedImages)
    return resizedImages


# rescales images from 32x32 to 1x1
def cifar10_color(X):
    print(f"resizing {len(X)} images: 32x32 -> 1x1")

    resizedImages = []
    for image in X:
        splitImage = np.split(image, 3)
        meanR = np.mean(splitImage[0]).astype('uint8')
        meanG = np.mean(splitImage[1]).astype('uint8')
        meanB = np.mean(splitImage[2]).astype('uint8')
        resizedImages.append([meanR, meanG, meanB])
    # convert array to ndarray
    resizedImages = np.asarray(resizedImages)
    return resizedImages


def cifar_10_bayes_learn(Xf, Y):
    print("learning in cifar_10_bayes_learn")

    # Dict where labels are keys and values are arrays with colours inside them
    # Arrays inside one key/label are equal to images pixel size * 3 (e.g. 2x2*3 = 12)
    #
    pixels = Xf[0].size // 3  # how many pixels in images, 1x1=1, 2x2=4, ...
    meanValues = {}
    for i in range(0, 10):
        meanValues[i] = []
        for j in range(0, pixels * 3):
            meanValues[i].append([])
    # add colours to correct labels
    for index, image in enumerate(Xf):
        label = Y[index]
        for x in range(0, pixels):
            meanValues[label][x].append(image[x])  # Red
            meanValues[label][x + pixels].append(image[x + pixels])  # Green
            meanValues[label][x + (2 * pixels)].append(image[x + (2 * pixels)])  # Blue

    # calculate covariances
    covariances = []
    for index in range(0, 10):
        valuesRGB = meanValues[index]
        valuesRGB = np.asarray(valuesRGB)
        covariances.append(np.cov(valuesRGB, dtype=np.longfloat))


    # calculate means and replace color arrays with them
    for label in range(0, 10):
        for x in range(0, pixels * 3):
            meanValues[label][x] = np.mean(meanValues[label][x])

    # priors, index = label
    labelFrequency = sorted((Counter(Y)).items())
    priors = []
    for label in labelFrequency:
        priors.append(label[1])
    priors = np.divide(priors, len(Xf))

    return meanValues, covariances, priors


def cifar10_classifier_bayes(x, mu, sigma, p):
    classProbabilities = []
    for i in range(0, 10):
        norm = sc.multivariate_normal.pdf(x, mu[i], sigma[i], allow_singular=True)
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
usedTestImages = 100  # 1-10000
usedTrainingImages = 1000

# resize images to 1x1 for Tasks 1&2
trainingImages1x1 = cifar10_color(trainingData[0:usedTrainingImages])
testImages1x1 = cifar10_color(testingData["data"][0:usedTestImages])

# parameters for naive bayes 1x1 and bayes 1x1
means, variances, priors = cifar_10_naivebayes_learn(trainingImages1x1, trainingLabels[0:usedTrainingImages])
meansB, covariancesB, priorsB = cifar_10_bayes_learn(trainingImages1x1, trainingLabels[0:usedTrainingImages])

# calculate predicted labels 1x1
predLabelsNaive1x1, predLabelsBayes1x1 = [], []
accuracyData = []  # for graph
for Image1x1 in testImages1x1:
    predLabelsNaive1x1.append(cifar10_classifier_naivebayes(Image1x1, means, variances, priors))
    predLabelsBayes1x1.append(cifar10_classifier_bayes(Image1x1, meansB, covariancesB, priorsB))

# calculate accuracies
accuracyNaive1x1 = class_acc(predLabelsNaive1x1, testingData["labels"][0:usedTestImages])
accuracyBayes1x1 = class_acc(predLabelsBayes1x1, testingData["labels"][0:usedTestImages])

print(f"1x1 bayes naive accuracy with {usedTestImages} test images: {accuracyNaive1x1}%")
print(f"1x1 bayes accuracy with {usedTestImages} test images: {accuracyBayes1x1}%")
accuracyData.append(accuracyBayes1x1)

# Task 3
tests = [2, 3, 4, 5, 6, 7, 8]  # test these pixel sizes. 2 mean 2x2 etc.
for W in tests:
    trainingImagesWxW = cifar10_WxW_color(trainingData[0:usedTrainingImages], W)
    testImagesWxW = cifar10_WxW_color(testingData["data"][0:usedTestImages], W)
    means, covariances, priors = cifar_10_bayes_learn(trainingImagesWxW, trainingLabels[0:usedTrainingImages])
    predLabels = []
    for testImage in testImagesWxW:
        predLabels.append(cifar10_classifier_bayes(testImage, means, covariances, priors))
    accuracyBayesWxW = class_acc(predLabels, testingData["labels"][0:usedTestImages])
    accuracyData.append(accuracyBayesWxW)
    print(f"{W}x{W} bayes accuracy with {usedTestImages} test images: {accuracyBayesWxW}%")

# draw  graph for accuracies
x = [1, 2, 3, 4, 5, 6, 7, 8]
customNames = ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6', '7x7', '8x8']
plt.xticks(x, customNames)
plt.plot(x, accuracyData, marker='o')
plt.show()




