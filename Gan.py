from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, BatchNormalization, Input, concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from keras.utils import to_categorical

import numpy as np
import os
import matplotlib.pyplot as plt

class GAN():
    def __init__(self):
        self.latent_dim = 100
        self.image_dim = 784
        self.label_dim = 10
        self.opt = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.d_loss_real_log = []
        self.d_loss_fake_log = []
        self.g_loss = []
        self.epoch = []
        self.labeled_data

        #Create and compile generator, used to generate images for training the discriminator
        self.generator = self.build_generator()
        self.generator.compile(optimizer = self.opt, loss = 'binary_crossentropy')

        #Create and compile discriminator, used to train discriminator indiviudally
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer = self.opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

        #Create and compile combined (generator,discriminator), used to train the generator on disciminator loss.
            ##ALL  labels must be of same input
        self.discriminator.trainable = False
        image = self.generator([self.noise, self.glabel])
        self.combined = Model([self.noise, self.glabel], self.discriminator([image, self.glabel]))
        self.combined.compile(optimizer = self.opt, loss = 'binary_crossentropy')

    def build_generator(self):
        model = Sequential([
            Dense(200),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(500),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(1000),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(self.image_dim, activation ='tanh')
		])
        self.noise = Input(shape=(self.latent_dim,))
        self.glabel = Input(shape=(self.label_dim,))
        model_input = concatenate([self.noise, self.glabel], 1)

        return Model([self.noise, self.glabel], model(model_input))
    
    def build_discriminator(self):
        model = Sequential([
            Dense(500, input_dim = self.image_dim + self.label_dim, activation = 'relu'),
            Dense(200, activation = 'relu'),
            Dense(100, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])
        self.image = Input(shape=(self.image_dim,))
        self.dlabel = Input(shape=(self.label_dim,))
        model_input = concatenate([self.image, self.dlabel], 1)

        return Model([self.image, self.dlabel], model(model_input))

    def get_samples(self, batch_size):
        indexes = np.random.randint(0, self.data.shape[0], batch_size)
        data = self.data[indexes]
        label =  self.labels[indexes]
        return data, label

    def train(self, epochs = 100, batch_size = 100, plot_interval = 10, sample_interval = 500):
        try:
            os.mkdir('images')
        except:
            pass
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        try:
            for epoch in range(epochs):
                #Generate images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                lab = np.random.randint(0, self.label_dim, (batch_size,))
                label = to_categorical(lab, num_classes= 10)

                fake_images = self.generator.predict([noise, label])
                #Get real images
                real_images, real_labels = self.get_samples(batch_size)
                #Train discriminator
                d_loss_fake = self.discriminator.train_on_batch([fake_images, label], fake)
                d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], real)
                d_loss_avg = np.add(d_loss_real, d_loss_fake) /2

                #Train generator
                g_loss = self.combined.train_on_batch([noise, label], real)
                print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, d_loss_avg[0], 100*d_loss_avg[1], g_loss))

                if (epoch % plot_interval == 0):
                    self.d_loss_fake_log.append(d_loss_avg[1])
                    self.d_loss_real_log.append(d_loss_avg[0])
                    self.g_loss.append(g_loss)
                    self.epoch.append(epoch)
                if (epoch % sample_interval == 0):
                    self.generate_images(epoch)
        except KeyboardInterrupt:
            self.save_generator_model()
        self.generate_images(epochs)

    def plot(self, name = 'test.png'):
        plt.plot(self.epoch, self.d_loss_real_log, label="Discriminator Loss - Real")
        plt.plot(self.epoch, self.d_loss_fake_log, label="Discriminator Loss - Fake")
        plt.plot(self.epoch, self.g_loss, label= "Generator Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN')
        plt.grid(True)
        plt.savefig("images/" + name)
        plt.show()

    def generate_images(self, generation):
        x, y = 3, 3
        noise = np.random.normal(0, 1, (x * y, self.latent_dim))
        lab = np.random.randint(0, self.label_dim, (x*y,))
        label = to_categorical(lab, num_classes= 10)
        images = self.generator.predict([noise, label])
        index = 0
        fig, axs = plt.subplots(x, y)
        for i in range(x):
            for j in range(y):
                axs[i,j].imshow(images[index, :784].reshape(28,28), cmap='gray')
                axs[i,j].axis('off')
                index += 1
        fig.savefig("images/%d.png" % generation)
        plt.close()
        
    def save_generator_model(self, name):
        self.generator.save('images/' + name + '.h5')

    def load_data(self, train, test):
        self.data = []
        self.labels = []
        data = []
        label = []
        with open(train, 'r') as f:
            print("Loading {}".format(train))
            header = True;
            for line in f:
                if (header):
                    header = False
                    continue
                image = line.split(',')
                data.append(list(map(int,image[1:])))
                lab = [0 for i in range(10)]
                lab[int(image[0])] = 1
                label.append(lab)
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
        self.data = np.array(data) / 127.5 - 1
        self.labels = np.array(label)
        print("Done Loading Data")

if __name__ == '__main__':
    gan = GAN()
    gan.load_data("data/train.csv", "data/test.csv")
    #gan.train(epochs = 10000, batch_size = 50, plot_interval = 50, sample_interval= 500)
    #gan.save_generator_model("generator")
    #gan.plot("gantest.png")
