import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('white')

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import autograd.numpy as np
import time
from IPython import display

def get_dataset(name):
    images, labels = None, None
    if name == 'ones_and_zeros':
        mnist = tfds.image_classification.MNIST()
        mnist.download_and_prepare()
        data = tfds.as_numpy(mnist.as_dataset(split='train', as_supervised=True).batch(100000))
        images, labels = next(iter(data))
        images, labels = images[labels <= 1][:, :, :, 0].astype(float) / 128 - 1., labels[labels <= 1].astype(float)
    elif name == 'horses_and_humans':
        hh = tfds.image_classification.HorsesOrHumans()
        hh.download_and_prepare()
        data = tfds.as_numpy(hh.as_dataset(split='train', as_supervised=True).map(lambda x, y: (tf.image.resize(
        tf.image.rgb_to_grayscale(x), (64, 64)), y)
        ).batch(100000))
        images, labels = next(iter(data))
        images, labels = images[labels <= 1][:, :, :, 0].astype(float) / 128 - 1., labels[labels <= 1].astype(float)
    elif name == 'cats_and_dogs':
        cd = tfds.image_classification.CatsVsDogs()
        cd.download_and_prepare()
        data = tfds.as_numpy(cd.as_dataset(split='train', as_supervised=True).map(lambda x, y: (tf.image.resize(
        tf.image.rgb_to_grayscale(x), (64, 64)), y)
        ).batch(100000))
        images, labels = next(iter(data))
        images, labels = images[labels <= 1][:, :, :, 0].astype(float) / 128 - 1., labels[labels <= 1].astype(float)

    f, subplots = plt.subplots(8, 8, figsize=(20, 20))
    i = 0
    for row in subplots:
        for subplot in row:
            subplot.imshow(images[i], cmap='gray')
            subplot.axis('off')
            i += 1
    plt.show()
    return images, labels

def gradient_descent(value_and_grad, w0, lr, steps, X, y, images):
    dims = np.array(images[0].shape).prod()

    f, ax = plt.subplots(X.shape[1] // dims, 3, figsize=(15,8))

    losses = []
    weights = w0
    for i in range(steps):
        loss, g = value_and_grad(weights, X, y)
        weights = weights - lr * g
        losses.append(loss)


        # Plotting code
        
        [a.cla() for a in ax.flatten()]
        [a.axis('off') for a in ax.flatten()[1:]]
        display.clear_output(wait =True)
        
        ax[0, 0].plot(losses)
        

        ax[0, 1].imshow(weights[:dims].reshape(images[0].shape))
        ax[0, 2].imshow(g[:dims].reshape(images[0].shape))
        ax[0, 1].set_title('Loss: %.3f, accuracy: %.3f' % (loss, accuracy(X, y, weights)))

        for j in range(1, ax.shape[0]):
            ax[j, 1].imshow(weights[(dims * j):(dims * (j+1))].reshape(images[0].shape))
            ax[j, 2].imshow(g[(dims * j):(dims * (j+1))].reshape(images[0].shape))
            ax[j, 0].imshow((X[0, (dims * j):(dims * (j+1))].reshape(images[0].shape)) )

        display.display(f)
        time.sleep(0.001)
        
    return weights, losses