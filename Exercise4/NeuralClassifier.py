import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.losses import CategoricalCrossentropy
from keras import callbacks, optimizers, preprocessing


# Stores 10 latest validation accuracies DURING training.
# Used to decrease lr if val_acc doesnt improve
class AccuracyHistory(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.acc = []  # Store validation accuracies here
        self.epochsSinceChange = 0

    def on_epoch_end(self, epoch, logs=None):
        # keep 10 LATEST validation accuracies in list. FIFO
        valAcc = logs.get('val_accuracy')
        self.acc.append(valAcc)
        self.acc = self.acc[-10:]

    def lrChanger(self, epoch, lr):
        # check if the biggest validation accuracy is in the latest 5 of 10 accuracies.
        # if not, lower learning rate.
        # epochsSinceChange is used to not allow lr change too fast
        if len(self.acc) != 10:
            return lr
        maxIndex = np.argmax(self.acc)
        print(f"Epochs since best val acc: {9-maxIndex}. learning rate: {lr}")
        if maxIndex < 5 and self.epochsSinceChange > 2:
            self.epochsSinceChange = 0
            print("DECREASING LEARNING RATE")
            return lr * 0.5
        else:
            self.epochsSinceChange += 1
            return lr


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
    # Take(remove) last 5% of training images and use them as validation images
    valShare = 0.05
    splitRatio = int(round(trImages.shape[0] * (1-valShare)))
    trImages, valImages = np.split(trImages, [splitRatio])
    trLabels, valLabels = np.split(trOneHot, [splitRatio])

    # Data augmentation, used with training images (trImages) only
    dataGen = preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        #vertical_flip=True,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #brightness_range=[0.5, 1.0],
    )

    # learning parameters
    numEpochs = 300  # might not matter because EarlyStop
    learningRate = 0.1  # decreased during training
    patience = 20
    batchSize = 32
    # Magic:
    #####################################################################################
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='sigmoid'))  # output layer
    #####################################################################################

    model.compile(optimizer=optimizers.SGD(lr=learningRate),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # fitting stopped early if X epochs in a row dont make val_loss smaller
    earlyStop = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
    accHis = AccuracyHistory()
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(accHis.lrChanger)
    hist = model.fit(dataGen.flow(trImages, trLabels, batch_size=batchSize, shuffle=True),
                     epochs=numEpochs,
                     validation_data=(valImages, valLabels),
                     callbacks=[earlyStop, lrScheduler, accHis],
                     verbose=1)

    # graphs for training/validation accuracy and loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.axhline(y=0.73, xmin=0, xmax=100, color='black')

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
usedTestImages = 10000  # 1-10000
usedTrainingImages = 50000
neural(trainingData[0:usedTrainingImages], trainingLabels[0:usedTrainingImages],
       testingData[0:usedTestImages], testingLabels[0:usedTestImages])

# neural(newTrainingData[0:10000], newTrainingLabels[0:10000],
#        newTestingData[0:2000], newTestingLabels[0:2000])
