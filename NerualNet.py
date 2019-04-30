
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, Dense, Dropout, BatchNormalization, Lambda, GRU, LSTM, Bidirectional, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
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
    def __init__(self):
        self.input_dimension = 784
        self.output_dimension = 10
        self.model = None
        self.trainData = []
        self.trainLabel = []
        self.testData = []
        self.testLabel = []
        self.opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
        self.build_model()
    
    def build_model(self):
        self.model = Sequential([
            Dense(128, input_dim = self.input_dimension, activation = 'relu'),
            Dense(self.output_dimension, activation = 'softmax')
        ])
        self.model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics=['accuracy'])

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
        self.plot(history.acc)
        score = self.model.evaluate(self.testData, self.testLabel)
        print(score)

    def plot(self, acc):
        plt.plot(range(len(acc)), acc, label="Model")
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Classifier')
        plt.grid(True)
        plt.show()
        
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
    nnclassifier = NerualNetwork()
    #nnclassifier.load_data(("trainimages", "trainlabels"), ("testimages", "testlabels"))
    nnclassifier.load_data("data/train.csv", "data/test.csv")
    nnclassifier.train(epochs=10, batch_size=100)