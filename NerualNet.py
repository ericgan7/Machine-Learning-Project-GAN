
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, Dense, Dropout, BatchNormalization, Lambda, GRU, LSTM, Bidirectional, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop, Nadam

import sys
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class NerualNetwork():
    def __init__(self):
        self.input_dimension = 784
        self.output_dimension = 10
        self.model = None
        self.trainData = []
        self.trainLabel = []
        self.testData = []
        self.testLabel = []
        self.opt = Adam(0.0002, 0.5, clipnorm=1)

    
        self.build_model()
    
    def build_model(self):
        self.model = Sequential([
            Dense(500, input_dim = self.input_dimension),
            LeakyReLU(alpha=0.2),
            Dense(self.output_dimension, activation = 'sigmoid')
        ])
        self.model.compile(optimizer = self.opt, loss = 'binary_crossentropy', metrics=['accuracy'])

    def load_data(self, train, test):
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
                lab = [0 for i in range(10)]
                lab[int(image[0])] = 1
                label.append(lab)
            self.trainData = np.array(data)
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
                lab = [0 for i in range(10)]
                lab[int(image[0])] = 1
                label.append(lab)
            self.testData = np.array(data)
            self.testLabel = np.array(label)
        print(self.trainData.shape)
        print(self.trainLabel.shape)
        print("Done Loading Data")

    def train(self, epochs = 100, batch_size = 32):
        print("Begin Training")
        self.model.fit(self.trainData, self.trainLabel, epochs = epochs, batch_size = batch_size)
        score = self.model.evaluate(self.testData, self.testLabel, batch_size = batch_size)
        print(score)

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
    use_saved = sys.argv[1] if len(sys.argv) > 1 else None
    nnclassifier = NerualNetwork(use_saved)
    #nnclassifier.load_data(("trainimages", "trainlabels"), ("testimages", "testlabels"))
    nnclassifier.load_data("data/train.csv", "data/test.csv")
    nnclassifier.train(epochs=10, batch_size=32)