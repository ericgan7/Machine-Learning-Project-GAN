from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

def sample_gen_images(num, generator):
    x, y = 3, 3
    for n in range(num):
        noise = np.random.normal(0, 1, (x * y, 100))
        lab = np.full((x*y,), n % 10)
        label = to_categorical(lab, num_classes= 10)
        images = generator.predict([noise, label])
        index = 0
        fig, axs = plt.subplots(x, y)
        for i in range(x):
            for j in range(y):
                axs[i,j].imshow(images[index, :784].reshape(28,28), cmap='gray')
                axs[i,j].axis('off')
                index += 1
        fig.savefig("eval/%d.png" % n)
        plt.close()

def evaluate_dist_generated(num, generator, evaluator):
    total = num * 10
    noise = np.random.normal(0, 1, (total, 100))
    lab = np.repeat(np.arange(10), num)
    label = to_categorical(lab, num_classes = 10)
    images = generator.predict([noise, label]).reshape(total, 28, 28, 1)
    labels = evaluator.predict(images)

    results = [[] for i in range(10)]
    for l, i in zip(labels, images):
        index = np.argmax(l)
        results[index].append(i)
    dist = [len(x)/total for x in results]
    plt.bar(np.arange(10), dist)
    plt.xlabel("Label")
    plt.ylabel("Percentag of samples")
    plt.title("Generated distribution")
    plt.show()

generator = load_model("eval/generator.h5")
evaluator = load_model("eval/convolutionNN.h5")
#evaluate_dist_generated(500, generator, evaluator)
sample_gen_images(50, generator)