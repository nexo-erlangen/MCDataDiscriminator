import os, sys, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utilities.generator import *
from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1_l2
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import ReLU, GlobalMaxPooling2D, Dropout, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D, Reshape, Concatenate
from keras.layers.convolutional import Convolution2D, Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten
from matplotlib import colors
from keras import layers
from functools import partial
from keras import backend as K #this was added by me, not sure if it is so working

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
# path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/GAN/'
#path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_01_24-11_45_21/GAN/'
# path_discriminator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/GAN/'
# path_save = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/'
path_load = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_06_14-15_13_29/GAN/'
path_save = path_load


def load_generator():
    import keras as ks
    generator = ks.models.load_model(path_load + 'generator-000.hdf5')
    # generator.load_weights(path_load + 'generator_weights-' + str(last_epoch) + '.hdf5')
    print '\nGenerator'
    generator.summary()
    return generator

def load_discriminator():
    import keras as ks
    discriminator = ks.models.load_model(path_load + 'discriminator-000.hdf5')
    # discriminator.load_weights(path_load + 'discriminator_weights-' + str(last_epoch) + '.hdf5')
    print '\nCritic'
    discriminator.summary()
    return discriminator

def plot_structure(model):
    import keras as ks
    try: # plot model, install missing packages with conda install if it throws a module error
        ks.utils.plot_model(model, to_file=path_save + 'plot_%s.png'%(str(model)),
                            show_shapes=True, show_layer_names=False)
    except OSError:
        print '\n\nCould not produce plot_%s.png'%(str(model))

generator = load_generator()
critic = load_discriminator()

models = [generator, critic]
for model in models:
    plot_structure(model)

