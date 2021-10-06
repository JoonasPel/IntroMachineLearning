import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D
import keras


def neural(trImages, trLabels, testImages, testLabels):
    # reshape vectors of 3072 to 32x32x3 images
    trImages = trImages.reshape(len(trImages), 3, 32, 32).transpose(0, 2, 3, 1)
    testImages = testImages.reshape(len(testImages), 3, 32, 32).transpose(0, 2, 3, 1)
    # scale values to 0-1
    trImages = trImages / 255.0
    testImages = testImages / 255.0
    # convert labels (0-9) to one-hots. e.g. 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    trOneHot = np.zeros((trLabels.size, np.unique(trLabels).size), dtype='float32')
    trOneHot[np.arange(trLabels.size), trLabels] = 1
    testOneHot = np.zeros((testLabels.size, np.unique(testLabels).size), dtype='float32')
    testOneHot[np.arange(testLabels.size), testLabels] = 1

    # define model and layers
    model = Sequential()
    numEpochs = 50
    lr = 0.04
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    # output layer
    model.add(Dense(2, activation='sigmoid'))

    opt = keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # fitting stopped early if X epochs in a row dont make val_loss smaller
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit(trImages, trOneHot, epochs=numEpochs, validation_split=0.05, shuffle=True,
                     callbacks=[earlyStop], verbose=1)

    # graphs for training/validation accuracy and loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.axhline(y=0.6, xmin=0, xmax=100, color='black')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # calculate accuracy with test images
    print("test data accuracy:")
    loss, acc = model.evaluate(testImages, testOneHot, verbose=2)


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

testingData = unpickle('cifar-10-batches-py/test_batch')["data"]
testingLabels = unpickle('cifar-10-batches-py/test_batch')["labels"]
testingLabels = np.asarray(testingLabels)

# New arrays with images with only label 0 or 1
label0, label1 = 0, 1
newTrainingData, newTrainingLabels, newTestingData, newTestingLabels = [], [], [], []
for idx, val in enumerate(trainingData):
    label = trainingLabels[idx]
    if(label == label0 or label == label1):
        newTrainingData.append(val)
        newTrainingLabels.append(label)
for idx, val in enumerate(testingData):
    label = testingLabels[idx]
    if(label == label0 or label == label1):
        newTestingData.append(val)
        newTestingLabels.append(label)
newTrainingData = np.asarray(newTrainingData)
newTrainingLabels = np.asarray(newTrainingLabels)
newTestingData = np.asarray(newTestingData)
newTestingLabels = np.asarray(newTestingLabels)


# Used data sizes
# usedTestImages = 10000  # 1-10000
# usedTrainingImages = 50000
# neural(trainingData[0:usedTrainingImages], trainingLabels[0:usedTrainingImages],
#        testingData[0:usedTestImages], testingLabels[0:usedTestImages])

neural(newTrainingData[0:10000], newTrainingLabels[0:10000],
       newTestingData[0:2000], newTestingLabels[0:2000])
