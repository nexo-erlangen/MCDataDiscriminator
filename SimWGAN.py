"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""

                                            ########################
                                            #                      #
                                            # Modified Version Sim #
                                            #                      #
                                            ########################



import argparse
import os, sys, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utilities.generator import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ReLU, GlobalMaxPooling2D, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.activations import linear
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from matplotlib import colors
from keras import regularizers
from keras import layers
import keras as ks



BATCH_SIZE = 20
batch_size = BATCH_SIZE
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # 10 in the paper

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"


######################
                    ##
status = 'test'    ##
                    ##
######################







if status == 'test':

    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + 'Dummy/'
    path_model = path_save

elif status == 'train':

    now = time.strftime("%Y_%m_%d-%H_%M_%S")
    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + str(now)
    os.mkdir(path_save)
    path_save = path_save + "/"
    path_model = path_save + "GAN"
    os.mkdir(path_model)
    path_model = path_model + "/"
else:
    raise ValueError('Error with status: %s'%status)


result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory
files = []
for i in range(15):
    files.append(path + str(i) + "-shuffled.hdf5")
fake = {'IsMC': [1]}
real = {'IsMC': [0]}


gen_fake = generate_batches_from_files(files, batch_size, wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)
gen_real = generate_batches_from_files(files, batch_size , wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=real, yield_mc_info=0)


MC_input, y_MC = gen_fake.next()
X_train, y_train = gen_real.next()

# files_test = []
# for i in range(15, no_files):
#     files_test.append(path + str(i) + "-shuffled.hdf5")
# gen_fake_test = generate_batches_from_files(files_test, batch_size, wires='V', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=fake, yield_mc_info=0)
# gen_real_test = generate_batches_from_files(files_test, batch_size , wires='V', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=real, yield_mc_info=0)
#
# MC_input_test, y_MC_test = gen_fake_test.next()
# X_test, y_test = gen_real_test.next()


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

# def refiner_network(input_image_tensor):
#     """
#     The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
#     than holistically modifying the image content, preserving the global structure and annotations.
#     :param input_image_tensor: Input tensor that corresponds to a synthetic image.
#     :return: Output tensor that corresponds to a refined synthetic image.
#     """
#     def resnet_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
#         """
#         A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
#         each with `nb_features` feature maps.
#         See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
#         :param input_features: Input tensor to ResNet block.
#         :return: Output tensor from ResNet block.
#         """
#         y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
#         y = layers.Activation('relu')(y)
#         y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(y)
#
#         y = layers.merge.add([input_features, y])
#         #y = layers.Activation('relu')(y)
#         return y
#
#     # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
#     x = layers.Convolution2D(128, 1, 1, border_mode='same', activation='relu')(input_image_tensor)
#
#     # the output is passed through 4 ResNet blocks
#     for _ in range(1):
#         x = resnet_block(x)
#
#     # the output of the last ResNet block is passed to a 1 x 1 convolutional layer producing 1 feature map
#     # corresponding to the refined synthetic image
#     return layers.Convolution2D(1, 1, 1, border_mode='same')(x)

def refiner_network(input_image_tensor):
    """
    The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.
        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(input_features)
        # y = BatchNormalization()(y)
        # y = layers.merge.add([y, input_features])
        # y = layers.Activation('relu')(y)
        return y

    # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
    # x_1 = layers.Convolution2D(128, 1, 1, border_mode='same', activation='relu')(input_image_tensor)
    # x_3 = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu')(input_image_tensor)
    # x_5 = layers.Convolution2D(128, 5, 5, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks

    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)

    # x = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform',
    #                kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = layers.merge.add([x, input_image_tensor])
    #
    # for _ in range(3):
    #     x = resnet_block(x, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3)
    #     x = resnet_block(x, nb_features=128, nb_kernel_rows=5, nb_kernel_cols=5)
    #     x = layers.Convolution2D(1, 1, 1, border_mode='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
    #     x = layers.merge.add([x, input_image_tensor])
    x = layers.Convolution2D(128, 7, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 1, 7, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 9, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 1, 9, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 11, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 1, 11, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 13, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Convolution2D(128, 1, 13, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)

    # x = layers.merge.add([input_image_tensor, x_1, x_3, x_5])
    # x = layers.Convolution2D(256, 3, 3, border_mode='same', activation='relu')(x)

    # x = layers.Convolution2D(1, 1, 1, border_mode='same')(x)

    # the output of the last ResNet block is passed to a 1 x 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image

    # x= Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    #
    #
    # y = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # y = BatchNormalization()(y)
    #
    # x = layers.merge.average([x, y])
    # x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = layers.merge.add([x, input_image_tensor])

    return x




# def refiner_network(input_image_tensor):
#     """
#     The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
#     than holistically modifying the image content, preserving the global structure and annotations.
#     :param input_image_tensor: Input tensor that corresponds to a synthetic image.
#     :return: Output tensor that corresponds to a refined synthetic image.
#     """
#     def resnet_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
#         """
#         A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
#         each with `nb_features` feature maps.
#         See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
#         :param input_features: Input tensor to ResNet block.
#         :return: Output tensor from ResNet block.
#         """
#         y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same', activation='relu', kernel_regularizer=regularizers.l2(1.e-2))(input_features)
#         # y = layers.Activation('relu')(y)
#         y = layers.Convolution2D(nb_features, nb_kernel_cols, nb_kernel_rows, border_mode='same', activation='relu', kernel_regularizer=regularizers.l2(1.e-2))(y)
#
#         # y = layers.merge.add([y, input_features])
#         # y = layers.Activation('relu')(y)
#         return y
#
#     # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
#     # x_1 = layers.Convolution2D(128, 1, 1, border_mode='same', activation='relu')(input_image_tensor)
#     # x_3 = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu')(input_image_tensor)
#     # x_5 = layers.Convolution2D(128, 5, 5, border_mode='same', activation='relu')(input_image_tensor)
#
#     # the output is passed through 4 ResNet blocks
#     # for _ in range(3):
#     # x = layers.Convolution2D(256, 3, 3, border_mode='same', activation='relu', kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
#     # x = resnet_block(input_image_tensor, nb_features=128, nb_kernel_rows=1, nb_kernel_cols=3)
#     # x = resnet_block(x, nb_features=128, nb_kernel_rows=1, nb_kernel_cols=5)
#     # x = layers.Convolution2D(1, 1, 1, border_mode='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
#
#     # x = layers.merge.add([input_image_tensor, x_1, x_3, x_5])
#     # x = layers.Convolution2D(256, 3, 3, border_mode='same', activation='relu')(x)
#
#     # x = layers.Convolution2D(1, 1, 1, border_mode='same')(x)
#
#     # the output of the last ResNet block is passed to a 1 x 1 convolutional layer producing 1 feature map
#     # corresponding to the refined synthetic image
#
#     x= Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
#     x = BatchNormalization()(x)
#     x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2))(x)
#     x = BatchNormalization()(x)
#
#
#     y = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
#     y = BatchNormalization()(y)
#
#     x = layers.merge.average([x, y])
#     x = Activation('relu')(x)
#     x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
#     return x
#
#
# def make_generator():
#     model = Sequential()
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2), input_shape=(38, 350, 1)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2)))
#     model.add(BatchNormalization())
#     # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#     #                  kernel_regularizer=regularizers.l2(1.e-2)))
#     # model.add(BatchNormalization())
#     # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#     #                  kernel_regularizer=regularizers.l2(1.e-2)))
#     # model.add(BatchNormalization())
#     # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#     #                  kernel_regularizer=regularizers.l2(1.e-2)))
#     # model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(1, (5, 5), padding='same'))
#     return model


def make_discriminator():
    model = Sequential()
    # model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2), input_shape=(38, 350, 1)))
    model.add(Conv2D(32, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(64, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    # model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(256, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    # model.add(MaxPooling2D(pool_size=(2, 4)))

    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())

    model.add(Dense(1))
    return model




class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])





discriminator = make_discriminator()


# Now we initialize the generator and discriminator.
# print('\nGenerator')
# synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
# generator = make_generator(synthetic_image_tensor)
# print(generator.summary())
#
# print('\nDiscriminator')
# discriminator = make_discriminator()
# print(discriminator.summary())
#
# print "Saving the structure in file"
# '''
# Save into file:
#     -Generator
#     -Discriminator
#     -GAN
# '''
# def save_in_file():
#     orig_stdout = sys.stdout
#     f = open(path_save + 'GAN_structure.txt', 'w')
#     sys.stdout = f
#
#     print('\nGenerator')
#     print(generator.summary())
#
#     print('\nDiscriminator')
#     print(discriminator.summary())
#
#     print 'Training ration:', TRAINING_RATIO   # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
#     print 'Gradient penality:', GRADIENT_PENALTY_WEIGHT   # 10 in the paper
#
#     sys.stdout = orig_stdout
#     f.close()
#
# save_in_file()

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
#####################
synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
refined_image_tensor = refiner_network(synthetic_image_tensor)
refined_or_real_image_tensor = layers.Input(shape=(38, 350, 1))
# discriminator_output = discriminator(refined_or_real_image_tensor)
discriminator_output = discriminator(refined_image_tensor)

#
# define models
#
generator_model_gen = Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner_gen')
# generator_model = Model(input=generator_model_test.input, output=discriminator_output, name='refiner_dis')
generator_model = Model(input=synthetic_image_tensor, output=discriminator_output, name='refiner')


# generator_model = Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')




# print(generator_model.summary())
# print(discriminator_model.summary())
'''
PRINTING
'''

print('\nGenerator')
print(generator_model.summary())

print('\nDiscriminator')
# discriminator = make_discriminator()
print(discriminator.summary())

print "Saving the structure in file"
'''
Save into file:
    -Generator
    -Discriminator
    -GAN
'''
def save_in_file():
    orig_stdout = sys.stdout
    f = open(path_save + 'GAN_structure.txt', 'w')
    sys.stdout = f

    print('\nGenerator')
    print(generator_model.summary())

    print('\nDiscriminator')
    print(discriminator.summary())

    print '\nTraining ration:', TRAINING_RATIO   # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
    print 'Gradient penality:', GRADIENT_PENALTY_WEIGHT   # 10 in the paper

    sys.stdout = orig_stdout
    f.close()

save_in_file()


######################

# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator_model.layers:
    layer.trainable = False
discriminator.trainable = True
generator_model.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=(38, 350, 1))
generator_input_for_discriminator = Input(shape=(38, 350, 1))
generated_samples_for_discriminator = refiner_network(generator_input_for_discriminator)
# print generated_samples_for_discriminator
# exit()
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])   #TODO add it back

# try: # plot model, install missing packages with conda install if it throws a module error
#     raise OSError
#     ks.utils.plot_model(discriminator_model, to_file=path_save + 'plot_model.png', show_shapes=True, show_layer_names=False)
# except OSError:
#     print 'could not produce plot_model.png ---- run generate_model_plot on CPU'

# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

def train_for_n(epochs=50, batch_size=batch_size):
    discriminator_loss = []
    generator_loss = []
    discriminator_loss_average = []
    generator_loss_average = []
    epoch_counter = []
    epoch_counter_fine = []
    for epoch in range(50):
        print 'epoch', str(epoch + 1), '/', epochs

        train_steps_per_epoch = int(getNumEvents(files) / batch_size / 2.)

        for i in range(train_steps_per_epoch):
            if i %100 == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")
                # if i != 0:
                #     print 'Discriminator accuracy', discriminator_acc[len(discriminator_acc) - 1]

            # np.random.shuffle(X_train)
            # np.random.shuffle(MC_input)
            # print("Epoch: ", epoch)
            # print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
            for j in range(TRAINING_RATIO):
                MC_input, y_MC = gen_fake.next()
                X_train, y_train = gen_real.next()
                discriminator_loss.append(discriminator_model.train_on_batch([X_train, MC_input],
                                                                             [positive_y, negative_y, dummy_y]))
                epoch_counter_fine.append(epoch+float(i+(float(j)/TRAINING_RATIO))/train_steps_per_epoch)

            # generator_loss.append(generator_model.train_on_batch(MC_input, positive_y)) # original: positive_y
            '''
            epoch_counter.append(epoch+float(i)/train_steps_per_epoch)
            # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
            if i * TRAINING_RATIO % 100 == 0:
                generator_loss_average.append(np.average(generator_loss, axis=0))
                generator_loss = []
            if i * TRAINING_RATIO % 100 == 0:
                discriminator_loss_average.append(np.average(discriminator_loss, axis=0))
                discriminator_loss = []
            '''
            generated_images = generator_model_gen.predict(MC_input)
            if i % 100 == 0:
                try:
                    if np.max(np.abs(generated_images[0])) > np.max(np.abs(MC_input[0])):
                        fmax = np.max(np.abs(generated_images[0]))
                    else:
                        fmax = np.max(np.abs(MC_input[0]))

                    plt.subplot(2, 1, 1)
                    plt.imshow(np.reshape(generated_images[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    plt.subplot(2, 1, 2)
                    plt.imshow(np.reshape(MC_input[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + '.pdf', bbox_inches='tight')
                    plt.close()
                    plt.clf()

                    for j in range(MC_input.shape[1]):
                        plt.subplot(2, 1, 1)
                        plt.plot(generated_images[0][j] + j * 20, color='black', linewidth=0.5)
                        plt.subplot(2, 1, 2)
                        plt.plot(MC_input[0][j] + j * 20, color='black', linewidth=0.5)
                    plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot.pdf', bbox_inches='tight')
                    plt.close()
                    plt.clf()

                    for j in range(MC_input.shape[1]):
                        plt.subplot(3, 1, 1)
                        plt.plot(generated_images[0][j] + j * 20, color='black', linewidth=0.5)
                        # plt.title('[Top] Modified MC, [Center] MC, [Bottom] ModMC-MC')
                        plt.ylabel('Modified MC')
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlim(0, 350)
                        plt.subplot(3, 1, 2)
                        plt.plot(MC_input[0][j] + j * 20, color='black', linewidth=0.5)
                        plt.ylabel('Original MC')
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlim(0, 350)
                        plt.subplot(3, 1, 3)
                        plt.plot(np.subtract(generated_images[0][j], MC_input[0][j]) + j * 20, color='black',
                                 linewidth=0.5)
                        plt.ylabel('Mod - Orig')
                        plt.xlabel('Time steps')
                        plt.yticks([])
                        plt.xlim(0, 350)
                    plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot_diff.pdf', bbox_inches='tight')
                    plt.close()
                    plt.clf()
                except:
                    print 'Plotting not successful:', epoch, i


                loss_discriminator = np.asarray(discriminator_loss)
                discriminator_loss_aver = discriminator_loss_average #= np.asarray(discriminator_loss_average)

                # generator_loss_aver = np.asarray(generator_loss_average)

                plt.plot(loss_discriminator[:, 0], color='red', markersize=12, label=r'Total')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_total.pdf')
                plt.close()
                plt.clf()

                plt.plot(loss_discriminator[:, 1] + loss_discriminator[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_wasserstein.pdf')
                plt.close()
                plt.clf()

                plt.plot(loss_discriminator[:, 3], color='royalblue', markersize=12, label=r'GradientPenalty', linestyle='dashed')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_gradientpenality.pdf')
                plt.close()
                plt.clf()

                plt.plot(generator_loss, label='generative loss')
                plt.legend()
                # plt.show()
                plt.savefig(path_save + 'loss_generative.pdf')
                plt.close()
                plt.clf()

                # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                np.savetxt(path_save + 'discriminator_loss.txt', (np.array(discriminator_loss_average)), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                # np.savetxt(path_save + 'generator_loss.txt', (np.array(generator_loss_average)), delimiter=',')

                # plt.plot(epoch_counter_fine, loss_discriminator[:, 0], color='red', markersize=12, label=r'Total')
                # plt.legend()
                # plt.savefig(path_save + 'loss_discriminitive_total.pdf')
                # plt.close()
                # plt.clf()
                #
                # plt.plot(epoch_counter_fine, loss_discriminator[:, 1] + loss_discriminator[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
                # plt.legend()
                # plt.savefig(path_save + 'loss_discriminitive_wasserstein.pdf')
                # plt.close()
                # plt.clf()
                #
                # plt.plot(epoch_counter_fine, loss_discriminator[:, 3], color='royalblue', markersize=12, label=r'GradientPenalty', linestyle='dashed')
                # plt.legend()
                # plt.savefig(path_save + 'loss_discriminitive_gradientpenality.pdf')
                # plt.close()
                # plt.clf()
                #
                # plt.plot(epoch_counter, generator_loss, label='generative loss')
                # plt.legend()
                # # plt.show()
                # plt.savefig(path_save + 'loss_generative.pdf')
                # plt.close()
                # plt.clf()
                #
                # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                # np.savetxt(path_save + 'discriminator_loss.txt', (np.array(discriminator_loss)), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                # np.savetxt(path_save + 'generator_loss.txt', (np.array(generator_loss)), delimiter=',')


        generator_model.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        generator_model_gen.save_weights(path_save + "GAN/generator_gen_weights-" + str(epoch) + ".hdf5")
        discriminator.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        print 'Saved weights'
        # - Plot the loss of discriminator and generator as function of iterations

generator_model.save(path_save + "GAN/generator-000" + ".hdf5")
generator_model_gen.save(path_save + "GAN/generator_gen-000" + ".hdf5")
discriminator.save(path_save + "GAN/discriminator-000" + ".hdf5")

generator_model.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
generator_model_gen.save_weights(path_save + "GAN/generator_gen_weights-000" + ".hdf5")
discriminator.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")

train_for_n(epochs=50, batch_size=batch_size)