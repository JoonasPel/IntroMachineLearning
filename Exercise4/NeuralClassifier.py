import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras

def neural(trImages, trLabels, testImages, testLabels):
    # reshape vectors of 3072 to 32x32x3 images
    trImages = trImages.reshape(len(trImages), 3, 32, 32).transpose(0, 2, 3, 1)
    testImages = testImages.reshape(len(testImages), 3, 32, 32).transpose(0, 2, 3, 1)
    # scale values to 0-1
    trImages = trImages / 255.0
    testImages = testImages / 255.0
    # convert labels (0-9) to one-hots. e.g. 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    labelsOneHot = np.zeros((trLabels.size, trLabels.max() + 1), dtype=int)
    labelsOneHot[np.arange(trLabels.size), trLabels] = 1

    # define model and layers
    #model = keras.Sequential()
    #model.add(layers.Dense(5, input_shape=(32, 32, 3), activation='relu'))
    # output layer
    #model.add(layers.Dense(10, activation='sigmoid'))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    hist = model.fit(trImages, trLabels, epochs=60)
    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    #opt = keras.optimizers.SGD(learning_rate=0.1)
    #model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    #training
    #usedEpochs = 1
    #model.fit(images, labelsOneHot, epochs=usedEpochs, verbose=1)

    #calculate accuracy with test images
    loss, acc = model.evaluate(testImages, testLabels, verbose=2)



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

# Used data sizes
usedTestImages = 100  # 1-10000
usedTrainingImages = 50000

neural(trainingData[0:usedTrainingImages], trainingLabels[0:usedTrainingImages],
       testingData[0:usedTestImages], testingLabels[0:usedTestImages])
