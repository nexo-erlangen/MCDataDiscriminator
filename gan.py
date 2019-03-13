import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1_l2
from keras.models import Sequential
import tensorflow as tf


def get_session(gpu_fraction=0.40):
    ''' Allocate only a fraction of the GPU RAM - (1080 GTX 8Gb)'''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def plot_images(images, figsize=(10, 10), fname=None):
    """ Plot some images """
    n_examples = len(images)
    dim = np.ceil(np.sqrt(n_examples))
    plt.figure(figsize=figsize)
    for i in range(n_examples):
        plt.subplot(dim, dim, i + 1)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap=plt.cm.Greys)
        plt.axis('off')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.close()


def make_trainable(model, trainable):
    """ Helper to freeze / unfreeze a model """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def build_generator(latent_dim):
    generator = Sequential(name='generator')
    generator.add(Dense(7 * 7 * 128, input_shape=(latent_dim,)))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(Reshape([7, 7, 128]))
    generator.add(UpSampling2D(size=(2,2)))
    generator.add(Conv2D(128, (5, 5), padding='same'))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(UpSampling2D(size=(2,2)))
    generator.add(Conv2D(64, (5, 5), padding='same'))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(Conv2D(1, (5, 5), padding='same', activation='sigmoid'))
    return generator


def build_discriminator(drop_rate=0.25):
    """ Discriminator network """
    discriminator = Sequential(name='discriminator')
    discriminator.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(28, 28, 1)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(drop_rate))
    discriminator.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(drop_rate))
    discriminator.add(Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(drop_rate))
    discriminator.add(Flatten())
    discriminator.add(Dense(256, activity_regularizer=l1_l2(1e-5)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(drop_rate))
    discriminator.add(Dense(2, activation='softmax'))
    return discriminator
