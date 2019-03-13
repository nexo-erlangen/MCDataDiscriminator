#how to read files, and use them as input


                                #######################
                                #                     #
                                # Modified Version v4 #
                                #                     #
                                #######################








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
from keras.layers import ReLU, GlobalMaxPooling2D, Dropout, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten
from matplotlib import colors
from keras import layers
from functools import partial
# from keras import backend as K #this was added by me, not sure if it is so working

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
batch_size=20
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = batch_size
NCR = 5

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
    raise ValueError('Error with status: %s' % status)

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

# MC_input, y_MC = gen_fake.next()
# X_train, y_train = gen_real.next()


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
    gradients = K.gradients(y_pred, averaged_batch)
    gradients_sqr_sum = K.sum(K.square(gradients)[0], axis=(1, 2, 3))
    gradient_penalty = penalty_weight * K.square(1 - K.sqrt(gradients_sqr_sum))
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors"""

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        super(_Merge, self).__init__(*args, **kwargs)

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def build_generator_graph(generator, critic):
    '''Builds the graph for training the generator part of the improved WGAN'''
    generator_in = Input(shape=(38, 350, 1))
    generator_out = generator([generator_in])
    critic_out = critic(generator_out)
    return Model(inputs=[generator_in], outputs=[critic_out])


def build_critic_graph(generator, critic, batch_size=1):
    '''Builds the graph for training the critic part of the improved WGAN'''
    generator_in_critic_training = Input(shape=(38, 350, 1), name="noise")
    shower_in_critic_training = Input(shape=(38, 350, 1), name='shower_maps')
    generator_out_critic_training = generator([generator_in_critic_training])
    out_critic_training_gen = critic(generator_out_critic_training)
    out_critic_training_shower = critic(shower_in_critic_training)
    averaged_batch = RandomWeightedAverage(batch_size, name='Average')([generator_out_critic_training, shower_in_critic_training])
    averaged_batch_out = critic(averaged_batch)
    return Model(inputs=[generator_in_critic_training, shower_in_critic_training], outputs=[out_critic_training_gen, out_critic_training_shower, averaged_batch_out]), averaged_batch


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












# build generator
# Feel free to modify the generator model
def build_generator():
    generator = Sequential(name='generator')
    generator.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(38, 350, 1)))
    generator.add(BatchNormalization())
    generator.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    generator.add(BatchNormalization())
    generator.add(Conv2D(128, (5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
    generator.add(BatchNormalization())
    generator.add(Conv2D(256, (5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
    generator.add(BatchNormalization())
    generator.add(Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    return generator


# build critic
# Feel free to modify the critic model
def build_critic():
    critic = Sequential(name='critic')
    critic.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(38, 350, 1)))
    critic.add(LeakyReLU())
    critic.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    critic.add(LeakyReLU())
    critic.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    critic.add(LeakyReLU())
    critic.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    critic.add(LeakyReLU())
    critic.add(GlobalMaxPooling2D())
    critic.add(Dense(100))
    critic.add(LeakyReLU())
    critic.add(Dense(1))
    return critic


generator = build_generator()
print(generator.summary())
critic = build_critic()
print(critic.summary())

# make trainings model for generator
make_trainable(critic, False)  # freeze the critic during the generator training
make_trainable(generator, True)  # unfreeze the generator during the generator training

generator_training = build_generator_graph(generator, critic)
generator_training.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss])

# make trainings model for critic
make_trainable(critic, True)  # unfreeze the critic during the critic training
make_trainable(generator, False)  # freeze the generator during the critic training

critic_training, averaged_batch = build_critic_graph(generator, critic, batch_size=BATCH_SIZE)
gradient_penalty = partial(gradient_penalty_loss, averaged_batch=averaged_batch, penalty_weight=GRADIENT_PENALTY_WEIGHT)  # construct the gradient penalty
gradient_penalty.__name__ = 'gradient_penalty'
critic_training.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss, wasserstein_loss, gradient_penalty])

# For Wassersteinloss
positive_y = np.ones(BATCH_SIZE)
negative_y = -positive_y
dummy = np.zeros(BATCH_SIZE)  # keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss

generator_loss = []
critic_loss = []

# trainings loop

def train_for_n(EPOCHS, BATCH_SIZE):
    iterations_per_epoch = int((getNumEvents(files) / 2. / BATCH_SIZE))
    for epoch in range(EPOCHS):

        print "epoch: ", epoch + 1, '/', EPOCHS

        for i in range(iterations_per_epoch):
            # if i * 20 % iterations_per_epoch == 0:
            if i % 100 == 0:
                print 'Step per epoch', i, '/', iterations_per_epoch, 'Time', time.strftime("%H:%M")



            for j in range(NCR):
                MC_input, y_MC = gen_fake.next()
                X_train, y_train = gen_real.next()
                noise = np.random.rand(batch_size, 38, 350, 1)
                critic_loss.append(critic_training.train_on_batch([MC_input, X_train],
                                                                  [negative_y, positive_y, dummy]))  # train the critic

            generator_loss.append(generator_training.train_on_batch([MC_input], [positive_y]))  # train the generator

            generated_images = generator.predict(MC_input)

            # if i * 20 % iterations_per_epoch == 0:
            if i %100 == 0:
                np.savetxt(path_save + 'discriminator_loss.txt', (critic_loss), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                np.savetxt(path_save + 'generator_loss.txt', (generator_loss), delimiter=',')

                # plot critic and generator loss
                plot_loss(critic_loss, name="critic")
                plot_loss(generator_loss, name="generator")

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
                    plt.plot(np.subtract(generated_images[0][j], MC_input[0][j]) + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Mod - Orig')
                    plt.xlabel('Time steps')
                    plt.yticks([])
                    plt.xlim(0, 350)
                plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot_diff.pdf', bbox_inches='tight')
                plt.close()
                plt.clf()

        # generator_predict.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        # critic_training.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        # generator_predict.save_weights(path_save + "GAN/GAN_weights-" + str(epoch) + ".hdf5")





print 'Calling train_for_n_epochs'

# generator_predict.save(path_save + "GAN/generator-000" + ".hdf5")
# critic_training.save(path_save + "GAN/discriminator-000" + ".hdf5")
# generator_training.save(path_save + "GAN/GAN-000.hdf5")
#
# generator_predict.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
# critic_training.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")
# generator_training.save_weights(path_save + "GAN/GAN_weights-000.hdf5")

train_for_n(EPOCHS=100, BATCH_SIZE=batch_size)

