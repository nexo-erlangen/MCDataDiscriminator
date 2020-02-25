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

import os, sys, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utilities.generator import *
import keras as ks
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
batch_size=20
GRADIENT_PENALTY_WEIGHT = 10

# NCR = 2


######################
                    ##
# status = 'test'   ##
status = 'train'  ##
                    ##
######################










if status == 'test':

    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + 'Dummy/'
    path_model = path_save + "GAN/"

elif status == 'train':

    now = time.strftime("%Y_%m_%d-%H_%M_%S")
    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + str(now)
    os.mkdir(path_save)
    path_save = path_save + "/"
    path_model = path_save + "GAN"
    os.mkdir(path_model)
    path_model = path_model + "/"
    print 'Folder created:', now
else:
    raise ValueError('Error with status: %s' % status)

result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory

files = []
for i in range(15):
    files.append(path + str(i) + "-shuffled.hdf5")
fake = {'IsMC': [1]}
real = {'IsMC': [0]}
gen_fake = generate_batches_from_files(files, batch_size, wires='UV', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)
gen_real = generate_batches_from_files(files, batch_size , wires='UV', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=real, yield_mc_info=0)

MC_input, y_MC = gen_fake.next()
X_train, y_train = gen_real.next()




def make_trainable(model, trainable):
    ''' Freezes/unfreezes the weights in the given model '''
    for layer in model.layers:
        # print(type(layer))
        if type(layer) is normalization.BatchNormalization:
            layer.trainable = True
        else:
            layer.trainable = trainable


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss - critic maximises the distance between its output for real and generated samples.
    To achieve this generated samples have the label -1 and real samples the label 1. Multiplying the outputs by the labels results to the wasserstein loss via the Kantorovich-Rubinstein duality"""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_batch, penalty_weight):
    """Calculates the gradient penalty (for details see arXiv:1704.00028v3).
    The 1-Lipschitz constraint for improved WGANs is enforced by adding a term to the loss which penalizes if the gradient norm in the critic unequal to 1"""

    gradients = []
    gradients.append(K.gradients(y_pred, averaged_batch[0]))
    gradients.append(K.gradients(y_pred, averaged_batch[1]))
    # print gradients
    gradients_sqr_sum_0 = K.sum(K.square(gradients[0])[0], axis=(1, 2, 3))
    gradients_sqr_sum_1 = K.sum(K.square(gradients[1])[0], axis=(1, 2, 3))

    gradient_penalty_0 = penalty_weight * K.square(1 - K.sqrt(gradients_sqr_sum_0))
    gradient_penalty_1 = penalty_weight * K.square(1 - K.sqrt(gradients_sqr_sum_1))

    gradient_penalty = K.concatenate((gradient_penalty_0, gradient_penalty_1))
    return K.mean(gradient_penalty)

    # gradients = K.gradients(y_pred, averaged_batch[1]) # [1] V_wires only
    # gradients_sqr_sum = K.sum(K.square(gradients)[0], axis=(1, 2, 3))
    # gradient_penalty = penalty_weight * K.square(1 - K.sqrt(gradients_sqr_sum))
    # return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors"""

    def __init__(self, batch_size, weights, *args, **kwargs):
        self.batch_size = batch_size
        self._weights = weights
        super(_Merge, self).__init__(*args, **kwargs)

    def _merge_function(self, inputs):
        # print len(inputs)
        # # weights = K.random_uniform((self.batch_size, 1, 1, 1))
        # averaged_batch = []
        # print inputs[0]
        # print inputs[1]
        # print inputs[2]
        # print inputs[3]
        # averaged_batch.append((weights * inputs[0]) + ((1 - weights) * inputs[1]))
        # averaged_batch.append((weights * inputs[2]) + ((1 - weights) * inputs[3]))
        # print len(averaged_batch), averaged_batch
        # return averaged_batch[0]

        # weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (self._weights * inputs[0]) + ((1 - self._weights) * inputs[1])


def build_generator_graph(generator, critic):
    '''Builds the graph for training the generator part of the improved WGAN'''
    generator_in = [Input(shape=(38, 350, 1), name="Wire_U"), Input(shape=(38, 350, 1), name="Wire_V")]
    generator_out = generator(generator_in)
    critic_out = critic(generator_out)
    return Model(inputs=(generator_in), outputs=[critic_out])


def build_critic_graph(generator, critic, batch_size=1):
    '''Builds the graph for training the critic part of the improved WGAN'''
    generator_in_critic_training_u = Input(shape=(38, 350, 1), name="noise_u")
    generator_in_critic_training_v = Input(shape=(38, 350, 1), name="noise_v")
    shower_in_critic_training_u = Input(shape=(38, 350, 1), name='shower_maps_u')
    shower_in_critic_training_v = Input(shape=(38, 350, 1), name='shower_maps_v')
    generator_out_critic_training = generator([generator_in_critic_training_u, generator_in_critic_training_v])
    # generator_out_critic_training = generator_in_critic_training
    out_critic_training_gen = critic(generator_out_critic_training)
    out_critic_training_shower = critic([shower_in_critic_training_u, shower_in_critic_training_v])
    averaged_batch = []
    # averaged_batch.append(RandomWeightedAverage(batch_size, name='Average_u')([generator_out_critic_training[0], shower_in_critic_training_u]))
    # averaged_batch.append(RandomWeightedAverage(batch_size, name='Average_v')([generator_out_critic_training[1], shower_in_critic_training_v]))
    weights = K.random_uniform((batch_size, 1, 1, 1))
    averaged_batch.append(RandomWeightedAverage(batch_size, weights, name='Average_u')([generator_out_critic_training[0], shower_in_critic_training_u]))
    averaged_batch.append(RandomWeightedAverage(batch_size, weights, name='Average_v')([generator_out_critic_training[1], shower_in_critic_training_v]))
    averaged_batch_out = critic(averaged_batch)
    return Model(inputs=[generator_in_critic_training_u, generator_in_critic_training_v,
                         shower_in_critic_training_u, shower_in_critic_training_v],
                 outputs=[out_critic_training_gen, out_critic_training_shower, averaged_batch_out]), averaged_batch


def plot_loss(loss, log_dir=".", name=""):
    """Plot the traings curve"""
    fig, ax1 = plt.subplots(1, figsize=(10, 4))
    epoch = np.arange(len(loss))
    loss = np.array(loss)
    try:
        plt.plot(epoch, loss[:, 0], color='red', markersize=12, label=r'Total')
        plt.plot(epoch, loss[:, 1] + loss[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
        plt.plot(epoch, loss[:, 3], color='royalblue', markersize=12, label=r'GradientPenalty', linestyle='dashed')
    except:
        plt.plot(epoch, loss[:], color='red', markersize=12, label=r'Total')

    plt.legend(loc='upper right', prop={'size': 10})
    ax1.set_xlabel(r'Iterations')
    ax1.set_ylabel(r'Loss')
    # ax1.set_ylim(np.min(loss)-0.5, 3.5)
    fig.savefig(path_save + '/%s_Loss.png' % name, dpi=120)
    plt.close('all')





def refiner_network(tensor_in):#u_tensor, v_tensor):
    """
    The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=32, nb_kernel_rows=1, nb_kernel_cols=5):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.
        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(input_features)

        y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(y)

        y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(y)
        # y = Conv2D(1, (1, 1), padding='same', activation='relu',
        #            kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-2))(input_features)
        #
        y = layers.merge.add([y, input_features])
        # y = layers.Add()([input_features, y])
        # y = layers.Activation('relu')(y)
        return y

    # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
    # x_1 = layers.Convolution2D(128, 1, 1, padding='same', activation='relu')(input_image_tensor)
    # x_3 = layers.Convolution2D(128, 3, 3, padding='same', activation='relu')(input_image_tensor)
    # x_5 = layers.Convolution2D(128, 5, 5, padding='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    # tensor_in = [u_tensor, v_tensor]
    tensor_out = []
    tensor_out.append(layers.Lambda(lambda x: x)(tensor_in[0]))
    for i in range(1, 2):
        x = Conv2D(32, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(tensor_in[i])

        x = Conv2D(32, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)

        x = Conv2D(32, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)

        y = Conv2D(32, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(tensor_in[i])

        x = layers.merge.add([y, x])

        for _ in range(7):
            x = resnet_block(x, 32, nb_kernel_rows=3)

        x1 = Conv2D(64, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)

        x1 = Conv2D(64, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x1)

        x1 = Conv2D(64, (3, 5), padding='same', activation='relu',
                    kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x1)

        y1 = Conv2D(64, (3, 5), padding='same', activation='relu',
                   kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)

        x = layers.merge.add([y1, x1])

        for _ in range(7):
            x = resnet_block(x, 64, nb_kernel_rows=3)

        tensor_out.append(Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2),
                                 name='generator_output_%s' %(str(i+1)))(x))

    return tensor_out



#
# def critic_inception(tensor_in):
#
#     def inception_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
#
#         x = Conv2D(64, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#                          kernel_regularizer=regularizers.l2(1.e-2))(input_features)
#         x = LeakyReLU()(x)
#
#         x = MaxPooling2D(pool_size=(2, 4))(x)
#
#         x = Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#                          kernel_regularizer=regularizers.l2(1.e-2))(x)
#         x = LeakyReLU()(x)
#         x = Flatten()(x)
#         return x
#
#     # x = inception_block(input_image_tensor, nb_features=128, nb_kernel_rows=7, nb_kernel_cols=1)
#     # x = inception_block(x, nb_features=128, nb_kernel_rows=9, nb_kernel_cols=1)
#     # x = inception_block(x, nb_features=128, nb_kernel_rows=11, nb_kernel_cols=1)
#     # x = inception_block(x, nb_features=128, nb_kernel_rows=13, nb_kernel_cols=1)
#     #
#     # tensor_in = [u_tensor, v_tensor]
#     tensor_out = []
#     for i in range(len(tensor_in)):
#         y1 = inception_block(tensor_in[i])
#         y2 = inception_block(tensor_in[i])
#         y3 = inception_block(tensor_in[i])
#         y4 = inception_block(tensor_in[i])
#         x = layers.merge.concatenate([y1, y2, y3, y4])
#
#         # x = Conv2D(64, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#         #                  kernel_regularizer=regularizers.l2(1.e-2))(tensor_in[i])
#         # x = LeakyReLU()(x)
#         #
#         # x = MaxPooling2D(pool_size=(2, 4))(x)
#         #
#         # x = Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#         #                  kernel_regularizer=regularizers.l2(1.e-2))(x)
#         # x = LeakyReLU()(x)
#         #
#         # x = Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#         #                  kernel_regularizer=regularizers.l2(1.e-2))(x)
#         # x = LeakyReLU()(x)
#         #
#         # x = Conv2D(256, (3, 5), padding='same', kernel_initializer='glorot_uniform',
#         #                  kernel_regularizer=regularizers.l2(1.e-2))(x)
#         # x = LeakyReLU()(x)
#         tensor_out.append(Dense(1)(x))
#         # tensor_out.append(Flatten()(x))
#     # x = GlobalAveragePooling2D()(x)
#
#     x = layers.concatenate(list(tensor_out))
#
#     x = Dense(1)(x)
#
#     return x

def critic_inception(tensor_in):#u_tensor, v_tensor):
    """
    The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def inception_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.
        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y1 = Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(input_features)

        y2 = Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(input_features)
        y2 = LeakyReLU()(y2)
        y2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(y2)

        y3 = Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(input_features)
        y3 = LeakyReLU()(y3)
        y3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(y3)
        y3 = LeakyReLU()(y3)
        y3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(y3)

        y4 = MaxPooling2D(pool_size=(1, 3), padding='same', strides=(1, 1))(input_features)
        y4 = LeakyReLU()(y4)
        y4 = Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(1.e-2))(y4)

        y1 = LeakyReLU()(y1)
        y2 = LeakyReLU()(y2)
        y3 = LeakyReLU()(y3)
        y4 = LeakyReLU()(y4)

        y = layers.merge.concatenate([y1, y2, y3, y4])
        # y = MaxPooling2D(pool_size=(1, 3))(y)
        # y = layers.Activation('relu')(y)
        return y

    # x = inception_block(input_image_tensor, nb_features=128, nb_kernel_rows=7, nb_kernel_cols=1)
    # x = inception_block(x, nb_features=128, nb_kernel_rows=9, nb_kernel_cols=1)
    # x = inception_block(x, nb_features=128, nb_kernel_rows=11, nb_kernel_cols=1)
    # x = inception_block(x, nb_features=128, nb_kernel_rows=13, nb_kernel_cols=1)
    #
    # tensor_in = [u_tensor, v_tensor]
    tensor_out = []
    for i in range(len(tensor_in)):
        x = Conv2D(64, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(1.e-2))(tensor_in[i])
        # # x = Conv2D(32, (3, 5), padding='same', kernel_initializer='glorot_uniform',
        # #                  kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)
        x = MaxPooling2D(pool_size=(2, 4))(x)
        #
        x = Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(1.e-2))(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)
        # # x = MaxPooling2D(pool_size=(2, 4))(x)
        #
        x = Conv2D(128, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(1.e-2))(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)
        # # x = MaxPooling2D(pool_size=(2, 4))(x)
        #
        x = Conv2D(256, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(1.e-2))(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)
        # # x = MaxPooling2D(pool_size=(2, 4))(x)

        tensor_out.append(Flatten()(x))
    # x = GlobalAveragePooling2D()(x)

    x = layers.concatenate(list(tensor_out))
    # x = Dense(5)(x)
    x = Dense(1)(x)

    return x




def create_model(model, layer_in, name=''):
    layer_out = model(layer_in)
    return Model(inputs=(layer_in), output=layer_out, name=name)

critic = create_model(model=critic_inception, layer_in=[Input(shape=(38, 350, 1), name='Wire_U'),
                                                        Input(shape=(38, 350, 1), name='Wire_V')], name='critic')
generator = create_model(model=refiner_network, layer_in=[Input(shape=(38, 350, 1), name='Wire_U'),
                                                        Input(shape=(38, 350, 1), name='Wire_V')], name='refiner')
generator.save(path_save + "GAN/generator-000" + ".hdf5")
critic.save(path_save + "GAN/discriminator-000" + ".hdf5")


def save(model, name, epoch, optimizer):
    if optimizer == True:
        ks.models.save_model(
            model,
            path_model + name + '_model-' + str(epoch) + '.hdf5',
            overwrite=True,
            include_optimizer=True)
        model.save_weights(path_model + name + '_weights-'+str(epoch)+'.hdf5')
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(path_model + name + '_optimizer-'+str(epoch)+'.pkl', 'wb') as f:
            pickle.dump(weight_values, f)
    else:
        ks.models.save_model(
            model,
            path_model + name + '_model-' + str(epoch) + '.hdf5',
            overwrite=True,
            include_optimizer=False)
        model.save_weights(path_model + name + '_weights-' + str(epoch) + '.hdf5')


print('\nGenerator')
print(generator.summary())
print('\nCritic')
print(critic.summary())

make_trainable(critic, False)  # freeze the critic during the generator training
make_trainable(generator, True)  # unfreeze the generator during the generator training

generator_training = build_generator_graph(generator=generator, critic=critic)
generator_training.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss])
#                                           GENERATOR


# make trainings model for critic
make_trainable(critic, True)  # unfreeze the critic during the critic training
make_trainable(generator, False)  # freeze the generator during the critic training

critic_training, averaged_batch = build_critic_graph(generator=generator, critic=critic, batch_size=batch_size)
gradient_penalty = partial(gradient_penalty_loss, averaged_batch=averaged_batch, penalty_weight=GRADIENT_PENALTY_WEIGHT)  # construct the gradient penalty
gradient_penalty.__name__ = 'gradient_penalty'
critic_training.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss, wasserstein_loss, gradient_penalty])
#                                       CRITIC



# plot_model(critic_training, to_file=log_dir + '/critic_training.png', show_shapes=True)

def plot_structure(model):
    try: # plot model, install missing packages with conda install if it throws a module error
        ks.utils.plot_model(model, to_file=path_save + 'GAN/plot_%s.png'%(str(model)),
                            show_shapes=True, show_layer_names=False)
    except OSError:
        print '\nCould not produce plot_%s.png'%(str(model))

models = [generator, generator_training, critic, critic_training]
for model in models:
    plot_structure(model)

# exit()

# For Wassersteinloss
positive_y = np.ones(batch_size)
negative_y = -positive_y
dummy = np.zeros(batch_size)  # keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss

generator_loss = []
critic_loss = []

def average_loss(loss):
    a = []
    for j in range(4):
        a.append(np.average(loss[:, j]))
    return a

def change_lr(model):
    lr = K.get_value(model.optimizer.lr) * 0.1
    K.set_value(model.optimizer.lr, lr)
    return


def train_for_n(EPOCHS, batch_size):
    iterations_per_epoch = int((getNumEvents(files) / batch_size / 2. ))
    NCR = 5
    # counter = 0
    for epoch in range(EPOCHS):

        print "epoch: ", epoch + 1, '/', EPOCHS

        # # Dinamic learning rate
        # if not epoch == 0:
        #     change_lr(critic_training)
        #     change_lr(generator_training)

        for i in range(iterations_per_epoch):

            # Dinamic NCR with iterations_per_epoch
            # if (i+iterations_per_epoch*epoch+1)%5000 == 0 and NCR < 10:
            #     NCR += 1
            #     print 'New Training ratio', NCR


            # Dinamic NCR with GradientPenalty
            # if (i+iterations_per_epoch*epoch+1)%1000 == 0 and NCR < 10 and (np.average(np.asarray(critic_loss)[-100:, -1])) < 30./NCR: #15.:
            #     NCR += 1
            #     print 'New Training ratio', NCR

            critic_loss_temp = []
            for j in range(NCR):
                MC_input, y_MC = gen_fake.next()
                X_train, y_train = gen_real.next()

                critic_loss_temp.append(critic_training.train_on_batch([MC_input[:,0,:,:,:], MC_input[:,1,:,:,:],
                                                                        X_train[:,0,:,:,:], X_train[:,1,:,:,:]],
                                                                        [negative_y, positive_y, dummy]))  # train the critic

            critic_loss.append(average_loss(np.asarray(critic_loss_temp)))
            generator_loss.append(generator_training.train_on_batch([MC_input[:,0,:,:,:], MC_input[:,1,:,:,:]],
                                                                    [positive_y]))  # train the generator


            if i % (iterations_per_epoch // 20) == 0:
            # if i % 100 == 0:
                print 'Step per epoch', i, '/', iterations_per_epoch, 'Time', time.strftime("%H:%M"), "Training ratio ", NCR

                np.savetxt(path_save + 'discriminator_loss.txt', (critic_loss), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                np.savetxt(path_save + 'generator_loss.txt', (generator_loss), delimiter=',')

                # plot critic and generator loss
                plot_loss(critic_loss, name="critic")
                plot_loss(generator_loss, name="generator")
                # print MC_input.shape # bs,2,38,350,1
                # print generated_images.shape # list len=2 bs,38,350,1

                # generated_images = generator_predict.predict(MC_input)
                generated_images = generator.predict([MC_input[:, 0, :, :, :], MC_input[:, 1, :, :, :]])

                generated_images[0] = np.reshape(generated_images[0], (batch_size, 38, 350))
                generated_images[1] = np.reshape(generated_images[1], (batch_size, 38, 350))
                MC_input = np.reshape(MC_input, (batch_size, 2, 38, 350))

                for j in range(MC_input.shape[2]):
                    plt.subplot(321)
                    plt.plot(MC_input[0,0][j] + j * 20, color='black', linewidth=0.5)
                    plt.title('U Wire')
                    plt.ylabel('Original MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(323)
                    plt.plot(generated_images[0][0][j] + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Modified MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(325)
                    plt.plot(np.subtract(generated_images[0][0][j], MC_input[0,0][j]) + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Mod - Orig')
                    plt.xlabel('Time steps')
                    plt.yticks([])
                    plt.xlim(0, 350)

                    plt.subplot(322)
                    plt.plot(MC_input[0,1][j] + j * 20, color='black', linewidth=0.5)
                    plt.title('V Wire')
                    # plt.ylabel('Modified MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(324)
                    plt.plot(generated_images[1][0][j] + j * 20, color='black', linewidth=0.5)
                    # plt.ylabel('Original MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(326)
                    plt.plot(np.subtract(generated_images[1][0][j], MC_input[0,1][j]) + j * 20, color='black', linewidth=0.5)
                    # plt.ylabel('Mod - Orig')
                    plt.xlabel('Time steps')
                    plt.yticks([])
                    plt.xlim(0, 350)
                plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot_diff.pdf', bbox_inches='tight')
                plt.close()
                plt.clf()

        # generator_predict.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        print 'Saving models and weights'
        save(generator, name='generator', epoch=epoch, optimizer=False)
        save(critic, name='discriminator', epoch=epoch, optimizer=False)
        save(generator_training, name='gen_train', epoch=epoch, optimizer=True)
        save(critic_training, name='disc_train', epoch=epoch, optimizer=True)




generator.save(path_save + "GAN/generator-000" + ".hdf5")
critic.save(path_save + "GAN/discriminator-000" + ".hdf5")
critic_training.save(path_save + "GAN/discr_train-000" + ".hdf5")
generator_training.save(path_save + "GAN/gen_train-000.hdf5")

print '\nSaving models'

save(generator, name='generator', epoch=000, optimizer=False)
save(critic, name='discriminator', epoch=000, optimizer=False)
save(generator_training, name='gen_train', epoch=000, optimizer=False)
save(critic_training, name='disc_train', epoch=000, optimizer=False)
# generator_predict.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
# critic_training.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")
# generator_training.save_weights(path_save + "GAN/GAN_weights-000.hdf5")



print '\nCalling train_for_n_epochs'
train_for_n(EPOCHS=100, batch_size=batch_size)