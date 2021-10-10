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
    def __init__(self):
        super().__init__()
        self.acc = []  # Store validation accuracies here
        self.epochs_since_change = 0

    def on_epoch_end(self, epoch, logs=None):
        # keep 10 LATEST validation accuracies in list. FIFO
        val_acc = logs.get('val_accuracy')
        self.acc.append(val_acc)
        self.acc = self.acc[-10:]

    def lr_changer(self, epoch, lr):
        # check if the biggest validation accuracy is in the latest 5 of 10 accuracies.
        # if not, lower learning rate.
        # epochsSinceChange is used to not allow lr change too fast
        if len(self.acc) != 10:
            return lr
        max_index = np.argmax(self.acc)
        print(f"Epochs since best val acc: {9 - max_index}. learning rate: {lr}")
        if max_index < 5 and self.epochs_since_change > 2:
            self.epochs_since_change = 0
            print("DECREASING LEARNING RATE")
            return lr * 0.5
        else:
            self.epochs_since_change += 1
            return lr


def neural(tr_images, tr_labels, test_images, test_labels):
    # reshape vectors of 3072 to 32x32x3 images
    tr_images = tr_images.reshape(len(tr_images), 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(len(test_images), 3, 32, 32).transpose(0, 2, 3, 1)
    # scale values to 0-1
    tr_images = tr_images / 255.0
    test_images = test_images / 255.0
    # convert labels (0-9) to one-hots. e.g. 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    tr_one_hot = np.zeros((tr_labels.size, np.unique(tr_labels).size), dtype='float32')
    tr_one_hot[np.arange(tr_labels.size), tr_labels] = 1
    test_one_hot = np.zeros((test_labels.size, np.unique(test_labels).size), dtype='float32')
    test_one_hot[np.arange(test_labels.size), test_labels] = 1
    # Take(remove) last 5% of training images and use them as validation images
    val_share = 0.05
    split_ratio = int(round(tr_images.shape[0] * (1 - val_share)))
    tr_images, val_images = np.split(tr_images, [split_ratio])
    tr_labels, val_labels = np.split(tr_one_hot, [split_ratio])

    # Data augmentation, used with training images (tr_images) only
    data_gen = preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    # learning parameters
    num_epochs = 300  # might not matter because EarlyStop
    learning_rate = 0.1  # decreased during training
    patience = 10
    batch_size = 32
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

    model.compile(optimizer=optimizers.SGD(lr=learning_rate),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # fitting stopped early if X epochs in a row don't make val_loss smaller
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
    acc_his = AccuracyHistory()
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(acc_his.lr_changer)
    hist = model.fit(data_gen.flow(tr_images, tr_labels, batch_size=batch_size, shuffle=True),
                     epochs=num_epochs,
                     validation_data=(val_images, val_labels),
                     callbacks=[early_stop, lr_scheduler, acc_his],
                     verbose=1)

    # graphs for training/validation accuracy and loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # calculate accuracy with test images
    # done with model.evaluate and also "manually" with class_acc.
    print("\ntest data accuracy with model.evaluate:")
    model.evaluate(test_images, test_one_hot, verbose=2)

    predicts_one_hot = model.predict(test_images)
    predict_labels = np.argmax(predicts_one_hot, axis=-1)
    accuracy = class_acc(predict_labels, test_labels)
    print(f"\ntest data accuracy with model.predict + class_acc:\n {accuracy}%")


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# returns classification accuracy(%) of provided labels
def class_acc(pred, qt):
    # check for empty or different sized lists, return -1 if error found
    if len(pred) == 0 or len(qt) == 0:
        return -1
    elif len(pred) != len(qt):
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
used_test_images = 10000  # 1-10000
used_training_images = 50000
neural(trainingData[0:used_training_images], trainingLabels[0:used_training_images],
       testingData[0:used_test_images], testingLabels[0:used_test_images])
