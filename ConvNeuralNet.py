from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import Callback

import numpy as np
import matplotlib.pyplot as plt

class AccuracyHistory(Callback):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def on_train_begin(self,logs={}):
        self.acc = []
        self.test = []

    def on_epoch_end(self,batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.test.append(self.model.evaluate(self.data, self.label)[1])

class NerualNetwork():
    def __init__(self, name = "temp"):
        self.name = name
        self.input_dimension = [28, 28, 1]
        self.output_dimension = 10
        self.model = None
        self.opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
        self.build_model()

    def build_model(self):
        self.model = Sequential([
            Conv2D(filters = 64, input_shape = self.input_dimension, kernel_size = 2, padding = 'same', activation = 'relu'),
            MaxPool2D(pool_size = 2),

            Conv2D(filters = 32, input_shape = self.input_dimension, kernel_size = 2, padding = 'same', activation = 'relu'),
            MaxPool2D(pool_size = 2),

            Flatten(),
            Dense(50, activation = 'relu', kernel_regularizer = l2(0.0)),
            Dense(self.output_dimension, activation = 'softmax')
        ])
        self.model.compile(optimizer = self.opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    def train(self, epochs = 100, batch_size = 32):
        print("Begin Training")
        history = AccuracyHistory(self.testData, self.testLabel)
        self.model.fit(self.trainData, self.trainLabel, epochs = epochs, batch_size = batch_size, callbacks=[history])
        #self.model.save("convolutionNN.h5")
        print("Training: {0}, Test {1}".format(history.acc[-1], history.test[-1]))
        self.plot(history.acc, history.test)

    def plot(self, acc, score):
        print("Training model with convolutional net")
        plt.figure()
        plt.plot(range(len(acc)), acc, label="Training Score")
        plt.plot(range(len(acc)), score, label = "Test Score")
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Classifier')
        plt.grid(True)
        plt.savefig("model/" + self.name + ".png")

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
            self.trainData = np.array(data)/127.5 - 1
            self.trainData = self.trainData.reshape([len(data)] + self.input_dimension)
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
            self.testData = np.array(data)/127.5 - 1
            self.testData = self.testData.reshape([len(data)] + self.input_dimension)
            self.testLabel = np.array(label)
        print("Done Loading Data")

if __name__ == '__main__':
    nnclassifier = NerualNetwork("ZeroCentered[64conv50relu+softmax]Adam0.001")
    nnclassifier.load_data("data/train.csv", "data/test.csv")
    nnclassifier.train(epochs= 10, batch_size= 100)

