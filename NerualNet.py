
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, Dense, Dropout, BatchNormalization, Lambda, GRU, LSTM, Bidirectional, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import Callback

import sys
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class AccuracyHistory(Callback):
    def on_train_begin(self,logs={}):
        self.acc = []

    def on_batch_end(self,batch, logs={}):
        self.acc.append(logs.get('acc'))

class NerualNetwork():
    def __init__(self, name = "temp", hidden_layer = 10, regularization = 0.0):
        self.name = name
        self.input_dimension = 784
        self.hidden_layer = hidden_layer
        self.regularization = regularization
        self.output_dimension = 10
        self.model = None
        self.opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
        self.build_model()
    
    def build_model(self):
        self.model = Sequential([
            Dense(self.hidden_layer, input_dim = self.input_dimension, activation = 'relu', kernel_regularizer = l1(self.regularization)),
            Dense(self.output_dimension, activation = 'softmax')
        ])
        self.model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics=['accuracy'])

    def load_data(self, train, test):
        self.trainData = []
        self.trainLabel = []
        self.testData = []
        self.testLabel = []
        with open(train, 'r') as f:
            print("Loading {}".format(train))
            header = True;
            data = []
            label = []
            for line in f:
                if (header):
                    header = False
                    continue
                image = line.split(',')
                data.append(list(map(int,image[1:])))
                #lab = [0 for i in range(10)]
                #lab[int(image[0])] = 1
                label.append(image[0])
            self.trainData = np.array(data) / 255
            self.trainLabel = np.array(label)
        with open(test, 'r') as f:
            print("Loading {}".format(test))
            header = True;
            data = []
            label = []
            for line in f:
                if (header):
                    header = False
                    continue
                image = line.split(',')
                data.append(list(map(int, image[1:])))
                #lab = [0 for i in range(10)]
                #lab[int(image[0])] = 1
                label.append(image[0])
            self.testData = np.array(data) / 255
            self.testLabel = np.array(label)
        print("Done Loading Data")

    def train(self, epochs = 100, batch_size = 32):
        print("Begin Training")
        history = AccuracyHistory()
        self.model.fit(self.trainData, self.trainLabel, epochs = epochs, batch_size = batch_size, callbacks=[history])
        score = self.model.evaluate(self.testData, self.testLabel)
        self.plot(history.acc, score[1])
        print(score)

    def plot(self, acc, score):
        print("Training model with {0} hidden layers".format(self.hidden_layer))
        plt.figure()
        plt.plot(range(len(acc)), acc, label="Training Score")
        plt.plot(range(len(acc)), np.full(len(acc), score), label = "Test Score")
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Classifier')
        plt.grid(True)
        plt.savefig("model/" + self.name + ".png")
        plt.figure("combined")
        plt.plot(range(len(acc)), acc, label = str(self.regularization))
        plt.plot(range(len(acc)), np.full(len(acc), score), label = str(self.regularization) + "score")

    def initcombinedplot(self):
        plt.figure("combined")
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Classifier')
        plt.grid(True)

    def savecombinedplot(self):
        plt.figure("combined")
        plt.legend()
        plt.savefig("model/combined.png")
        
"""
    def load_data(self, train, test):
        self.load(train[0], self.input_dimension, self.trainData)
        self.load(train[1], 1, self.trainLabel)
        self.load(test[0], self.input_dimension, self.testData)
        self.load(test[1], 1, self.testLabel)

    def load(self, fileName, dataSize, saveTo):
        with open("data/" + fileName, 'rb') as f:
            data = []
            while True:
                sample = f.read(dataSize)
                if len(sample) == 0:
                    break
                data.append([b for b in sample])
        saveTo = np.array(data)
        print(saveTo.shape)
"""
if __name__ == '__main__':
    sizes = [0.01, 0.05, 0.1, 0.2 ,0.3, 0.5]
    nnclassifier = NerualNetwork("[temprelu+softmax]Adam0.001", 10, 0.1)
    nnclassifier.load_data("data/train.csv", "data/test.csv")
    nnclassifier.initcombinedplot()
    for i in sizes:
        nnclassifier.__init__("[100relu+softmax]Adam0.001+l1" + str(i), 100, i)
        #nnclassifier.load_data(("trainimages", "trainlabels"), ("testimages", "testlabels"))
        nnclassifier.train(epochs=10, batch_size=500)
    nnclassifier.savecombinedplot()