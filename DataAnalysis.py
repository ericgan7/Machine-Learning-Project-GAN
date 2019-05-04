import numpy as np
import matplotlib.pyplot as plt

class DataAnalysis():
    def __init__(self, train, test):
        self.label_dim = 10
        self.data = [[] for i in range(self.label_dim)]

        with open(train, 'r') as f:
            print("Loading {}".format(train))
            header = True;
            for line in f:
                if (header):
                    header = False
                    continue
                image = line.split(',')
                label = int(image[0])
                self.data[label].append(list(map(int,image[1:])))
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
                label = int(image[0])
                self.data[label].append(list(map(int,image[1:])))
        for label in range(self.label_dim):
            self.data[label] = np.array(self.data[label])
        print("Done Loading Data")

    def get_image(self, label, num):
        indexes = np.random.randint(0, len(self.data[label]), num)
        return self.data[label][indexes]
    
    def plot_samples(self, num):
        x, y = 3, 3
        for iteration in range(num):
            fig, axs = plt.subplots(x, y)
            images = self.get_image(iteration % self.label_dim, x * y)
            index = 0
            for i in range(x):
                for j in range(y):
                    axs[i,j].imshow(images[index, :784].reshape(28,28), cmap='gray')
                    axs[i,j].axis('off')
                    index += 1
            fig.savefig("samples/%d.png" % iteration)
            plt.close()

if __name__ == '__main__':
    da = DataAnalysis("data/train.csv", "data/test.csv")
    da.plot_samples(100)