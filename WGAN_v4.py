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
from keras import backend as K #this was added by me, not sure if it is so working

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

MC_input, y_MC = gen_fake.next()
X_train, y_train = gen_real.next()


def make_trainable(model, trainable):
    ''' Freezes/unfreezes the weights in the given model '''
    for layer in model.layers:
        #print(type(layer))
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

def build_generator_predict(generator):
    '''Builds the graph for training the generator part of the improved WGAN'''
    generator_in = Input(shape=(38, 350, 1))
    generator_out = generator([generator_in])
    return Model(inputs=[generator_in], outputs=[generator_out])

def build_critic_graph(generator, critic, batch_size=1):
    '''Builds the graph for training the critic part of the improved WGAN'''
    generator_in_critic_training = layers.Input(shape=(38, 350, 1), name="noise")
    shower_in_critic_training = Input(shape=(38, 350, 1), name='shower_maps')
    generator_out_critic_training = generator(generator_in_critic_training)
    # generator_out_critic_training = refiner_network(generator_in_critic_training)
    out_critic_training_gen = critic(generator_out_critic_training)
    out_critic_training_shower = critic(shower_in_critic_training)
    averaged_batch = RandomWeightedAverage(batch_size, name='Average')([generator_out_critic_training, shower_in_critic_training])
    averaged_batch_out = critic(averaged_batch)
    return Model(inputs=[generator_in_critic_training, shower_in_critic_training], outputs=[out_critic_training_gen, out_critic_training_shower, averaged_batch_out]), averaged_batch


def plot_loss(loss, name=""):
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
    generator.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    generator.add(BatchNormalization())
    generator.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    generator.add(BatchNormalization())
    generator.add(Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
    return generator


#REFINER

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
        y = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(input_features)

        y = Conv2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(y)

        y = layers.merge.add([y, input_features])
        return y

    # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
    # x_1 = layers.Convolution2D(128, 1, 1, border_mode='same', activation='relu')(input_image_tensor)
    # x_3 = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu')(input_image_tensor)
    # x_5 = layers.Convolution2D(128, 5, 5, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    # x = resnet_block(input_image_tensor)
    # for _ in range(3):
    #     x = resnet_block(x)
    x = layers.Conv2D(128, 7, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 1, 7, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 9, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 1, 9, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 11, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 1, 11, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 13, 1, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    x = layers.Conv2D(128, 1, 13, border_mode='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1.e-2))(x)
    x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = resnet_block(x, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3)
    # x = resnet_block(x, nb_features=128, nb_kernel_rows=5, nb_kernel_cols=5)
    # x = layers.Convolution2D(1, 1, 1, border_mode='same', kernel_regularizer=regularizers.l2(1.e-2))(x)

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
    x = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = layers.merge.add([x, input_image_tensor])

    return x


# def refiner_network(input_image_tensor):
#
#     def resnet_block(input_features, nb_features=128, nb_kernel_rows=3, nb_kernel_cols=3):
#         y = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                    kernel_regularizer=regularizers.l2(1.e-2))(input_features)
#
#         y = Conv2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                    kernel_regularizer=regularizers.l2(1.e-2))(y)
#
#         y = layers.merge.add([y, input_features])
#         return y
#
#     x = resnet_block(input_image_tensor)
#     for _ in range(3):
#         x = resnet_block(x)
#
#     x = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-2))(x)
#
#     return x





# build critic
# Feel free to modify the critic model
def build_critic():
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


generator = build_generator()
print(generator.summary())
critic = build_critic()
print(critic.summary())

# make trainings model for generator
make_trainable(critic, False)  # freeze the critic during the generator training
make_trainable(generator, True)  # unfreeze the generator during the generator training

generator_training = build_generator_graph(generator, critic)
'''
synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
refined_image_tensor = refiner_network(synthetic_image_tensor)
refined_or_real_image_tensor = layers.Input(shape=(38, 350, 1))
# discriminator_output = discriminator(refined_or_real_image_tensor)
discriminator_output = critic(refined_image_tensor)

generator_predict = Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner_gen')
generator_training = Model(input=synthetic_image_tensor, output=discriminator_output, name='refiner')


make_trainable(critic, False)  # unfreeze the critic during the critic training
#make_trainable(generator_training, True)  # freeze the generator during the critic training
# make_trainable(generator_predict, True)  # freeze the generator during the critic training
'''
generator_training.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss])

# make_trainable(critic, False)  # unfreeze the critic during the critic training
# #make_trainable(generator_training, True)  # freeze the generator during the critic training
# make_trainable(generator_predict, True)  # freeze the generator during the critic training
#
# generator_training.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss])
#plot_model(generator_training, to_file=log_dir + '/generator_training.png', show_shapes=True)

# generator_predict = build_generator_predict(generator)



critic_training, averaged_batch = build_critic_graph(generator=generator, critic=critic, batch_size=BATCH_SIZE)
gradient_penalty = partial(gradient_penalty_loss, averaged_batch=averaged_batch, penalty_weight=GRADIENT_PENALTY_WEIGHT)  # construct the gradient penalty
gradient_penalty.__name__ = 'gradient_penalty'

# make trainings model for critic
make_trainable(critic, True)  # unfreeze the critic during the critic training
make_trainable(generator, False)
# make_trainable(generator_training, False)  # freeze the generator during the critic training
# make_trainable(generator_predict, False)  # freeze the generator during the critic training

critic_training.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss, wasserstein_loss, gradient_penalty])
#plot_model(critic_training, to_file=log_dir + '/critic_training.png', show_shapes=True)







# For Wassersteinloss
positive_y = np.ones(BATCH_SIZE)
negative_y = -positive_y
dummy = np.zeros(BATCH_SIZE)  # keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss

generator_loss = []
critic_loss = []
critic_prediction = []

def train_for_n(EPOCHS, BATCH_SIZE):
    iterations_per_epoch = int((getNumEvents(files) / 2. / BATCH_SIZE))
    for epoch in range(EPOCHS):

        print "epoch: ", epoch + 1, '/', EPOCHS

        for i in range(iterations_per_epoch):
            if i * 20 % iterations_per_epoch == 0:
                print 'Step per epoch', i, '/', iterations_per_epoch, 'Time', time.strftime("%H:%M")



            for j in range(NCR):
                MC_input, y_MC = gen_fake.next()
                X_train, y_train = gen_real.next()

                critic_loss.append(critic_training.train_on_batch([MC_input, X_train],
                                                                  [negative_y, positive_y, dummy]))  # train the critic

            # generator_loss.append(generator_training.train_on_batch([MC_input], [positive_y]))  # train the generator

            # generated_images = generator_predict.predict(MC_input)
            generated_images = generator.predict(MC_input)

            if i * 20 % iterations_per_epoch == 0:
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
        critic_training.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        # generator_predict.save_weights(path_save + "GAN/GAN_weights-" + str(epoch) + ".hdf5")
        generator.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")





print 'Calling train_for_n_epochs'

# generator_predict.save(path_save + "GAN/generator-000" + ".hdf5")
# critic_training.save(path_save + "GAN/discriminator-000" + ".hdf5")
# generator_training.save(path_save + "GAN/GAN-000.hdf5")
#
# generator_predict.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
# critic_training.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")
# generator_training.save_weights(path_save + "GAN/GAN_weights-000.hdf5")

train_for_n(EPOCHS=100, BATCH_SIZE=batch_size)





















'''

print('\nGenerator')
print(generator.summary())

print('\nDiscriminator')
print(discriminator.summary())

print('\nGenerative Adversarial Network')
print(GAN.summary())


print "Saving the structure in file"
'''
# Save into file:
#     -Generator
#     -Discriminator
#     -GAN
'''
def save_in_file():
    orig_stdout = sys.stdout
    f = open(path_save + 'GAN_structure.txt', 'w')
    sys.stdout = f

    print('\nGenerator')
    print(generator.summary())

    print('\nDiscriminator')
    print(discriminator.summary())

    print('\nGenerative Adversarial Network')
    print(GAN.summary())

    sys.stdout = orig_stdout
    f.close()

save_in_file()

'''

# main training loop
def train_for_n(epochs=50, batch_size=32):
    discriminator_loss = []
    generator_loss = []
    discriminator_loss_average = []
    generator_loss_average = []
    epoch_counter = []
    epoch_counter_fine = []
    losses_mean = {"d": [], "g": []}
    discriminator_acc_mean = []
    losses = {"d": [], "g": []}
    discriminator_acc = []

    for epoch in range(epochs):
        print 'epoch',str(epoch+1),'/',epochs
        # if epoch != 0:
        #     print 'Discriminator accuracy', discriminator_acc[len(discriminator_acc) - 1]#, len(discriminator_acc)

        train_steps_per_epoch = int(getNumEvents(files) / batch_size / 2.)

        for i in range(train_steps_per_epoch):
            if i % 100 == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")
                # if i != 0:
                #     print 'Discriminator accuracy', discriminator_acc[len(discriminator_acc) - 1]
            # if i % train_steps_per_epoch/float(3750) == 0:
            #     print 'minibatch %i of %i'%(i, train_steps_per_epoch)
            # Create a mini-batch of data (X: real images + fake images, y: corresponding class vectors)
            MC_input, y_MC = gen_fake.next()
            X_train, y_train = gen_real.next()

            # print xs_i.shape
            # print ys_i.shape
            # noise_mask = ys_i[:, 1] == 1
            # real_mask = np.invert(noise_mask)
            # noise_gen = xs_i[noise_mask]
            # real_gen = xs_i[real_mask]
            generated_images = generator.predict(MC_input)  # generated images
            # if i % 100 == 0:
            #     plt.subplot(2, 1, 1)
            #     plt.imshow(np.reshape(generated_images[0], (38, 350)), aspect='auto')
            #     plt.subplot(2, 1, 2)
            #     plt.imshow(np.reshape(MC_input[0], (38, 350)), aspect='auto')
            #     plt.show()
            # print 'Generator prediction sahape', generated_images.shape
            #            if i == 0:
            #                plt.imshow(np.reshape(generated_images[0], (76, 350)), aspect='auto')
            #                plt.show()

            # print 'Iteration', str(i+1),'/',iterations_per_epoch
            X = np.concatenate((X_train, generated_images))
            y = np.concatenate((y_train, y_MC))  # class vector
            # y[0:xs_i[real_mask].shape[0], 1] = 1
            # y[generated_images.shape[0]:, 0] = 1

            # Train the discriminator on the mini-batch
            # print 'Train the discriminator on the mini-batch'
            # for _ in range(5):
            d_loss, d_acc = discriminator.train_on_batch(X, y)
            losses["d"].append(d_loss)
            discriminator_acc.append(d_acc)

            # Create a mini-batch of data (X: noise, y: class vectors pretending that these produce real images)
            # noise_tr = MC_input[i * batch_size:(i + 1) * batch_size, :, :, :]
            #            for i in range(10):
            #                plt.imshow(np.reshape(noise_tr[i], (76, 350)), aspect='auto')
            #                plt.show()

            # y2 = np.zeros([batch_size, 2])
            # y2[:, 1] = 1
            # for _ in range(5):
            g_loss = GAN.train_on_batch(MC_input, y_train) #original y_MC
            losses["g"].append(g_loss)

            #TODO test on batch

            # if len(discriminator_acc) >= train_steps_per_epoch//20:
            #     discriminator_acc_mean.append(np.mean(discriminator_acc))
            #     discriminator_acc = []
            #     losses_mean["d"].append(np.mean(losses["d"]))
            #     losses_mean["g"].append(np.mean(losses["g"]))
            #     losses = {"d": [], "g": []}

            # if i % 100 == 0:
            if i * 20 % train_steps_per_epoch == 0:
                discriminator_acc_mean.append(np.mean(discriminator_acc))
                discriminator_acc = []
                losses_mean["d"].append(np.mean(losses["d"]))
                losses_mean["g"].append(np.mean(losses["g"]))
                losses = {"d": [], "g": []}
                try:
                    if np.max(np.abs(generated_images[0])) > np.max(np.abs(MC_input[0])):
                        fmax = np.max(np.abs(generated_images[0]))
                    else:
                        fmax = np.max(np.abs(MC_input[0]))

                    plt.subplot(2, 1, 1)
                    plt.imshow(np.reshape(generated_images[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    plt.title('[Top] Modified MC, [Bottom] MC')
                    plt.subplot(2, 1, 2)
                    plt.imshow(np.reshape(MC_input[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + '.pdf', bbox_inches='tight')
                    plt.close()
                    plt.clf()

                    for j in range(MC_input.shape[1]):
                        plt.subplot(2, 1, 1)
                        plt.plot(generated_images[0][j] + j * 20, color='black', linewidth=0.5)
                        #plt.title('[Top] Modified MC, [Bottom] MC')
                        plt.ylabel('Modified MC')
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlim(0, 350)
                        plt.subplot(2, 1, 2)
                        plt.plot(MC_input[0][j] + j * 20, color='black', linewidth=0.5)
                        plt.ylabel('Original MC')
                        plt.xlabel('Time steps')
                        plt.yticks([])
                        plt.xlim(0, 350)
                    plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot.pdf', bbox_inches='tight')
                    plt.close()
                    plt.clf()

                    for j in range(MC_input.shape[1]):
                        plt.subplot(3, 1, 1)
                        plt.plot(generated_images[0][j] + j * 20, color='black', linewidth=0.5)
                        #plt.title('[Top] Modified MC, [Center] MC, [Bottom] ModMC-MC')
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
                except:
                    print 'Plotting not successful:', epoch, i

                #loss_discriminator = np.asarray(discriminator_loss)
                # discriminator_loss_aver = np.asarray(discriminator_loss_average)
                # generator_loss_aver = np.asarray(generator_loss_average)
                try:
                    plt.plot(losses_mean["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.7)
                    plt.plot(losses_mean["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                    plt.title('Loss')
                    plt.legend()
                    plt.xlabel('Training Steps')
                    plt.savefig(path_save + 'loss.pdf')
                    plt.close()
                    plt.clf()

                    plt.semilogy(losses_mean["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.7)
                    plt.semilogy(losses_mean["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                    plt.title('Loss')
                    plt.legend()
                    plt.xlabel('Training Steps')
                    plt.savefig(path_save + 'loss_log.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc_mean, color='green', label='discriminator accuracy', marker='.', linestyle='None', markersize=1)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    #plt.legend()
                    plt.savefig(path_save + 'discriminator_acc_no-line.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc_mean, color='green', label='discriminator accuracy', linewidth=0.5)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    #plt.legend()
                    plt.savefig(path_save + 'discriminator_acc.pdf')
                    plt.close()
                    plt.clf()

                    # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                    np.savetxt(path_save + 'discriminator_loss.txt', (losses_mean["d"]), delimiter=',')
                    # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                    np.savetxt(path_save + 'generator_loss.txt', (losses_mean["g"]), delimiter=',')
                    np.savetxt(path_save + 'discriminator_acc.txt', (discriminator_acc_mean), delimiter=',')
                except:

                    print 'Plotting and saving normal loss and accuracy, mean not aviable yet'

                    plt.plot(losses["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.7)
                    plt.plot(losses["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                    plt.title('Loss')
                    plt.legend()
                    plt.xlabel('Training Steps')
                    plt.savefig(path_save + 'loss.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc, color='green', label='discriminator accuracy', marker='.',
                             linestyle='None', markersize=1)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    # plt.legend()
                    plt.savefig(path_save + 'discriminator_acc_no-line.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc, color='green', label='discriminator accuracy', linewidth=0.5)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    # plt.legend()
                    plt.savefig(path_save + 'discriminator_acc.pdf')
                    plt.close()
                    plt.clf()

                    # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                    np.savetxt(path_save + 'discriminator_loss.txt', (losses["d"]), delimiter=',')
                    # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                    np.savetxt(path_save + 'generator_loss.txt', (losses["g"]), delimiter=',')
                    np.savetxt(path_save + 'discriminator_acc.txt', (discriminator_acc), delimiter=',')

        generator.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        discriminator.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        GAN.save_weights(path_save + "GAN/GAN_weights-" + str(epoch) + ".hdf5")
        print 'Saved weights'


#         for j in range(5):
#             MC_input, X_test, X_train = create_batch(j)
#             iterations_per_epoch = X_train.shape[0]//batch_size    # number of training steps per epoch
#             #perm = np.random.choice(320, size=320, replace='False')
#
#             for i in range(iterations_per_epoch):
#
#                 # Create a mini-batch of data (X: real images + fake images, y: corresponding class vectors)
#                 image_batch = X_train[i*batch_size:(i+1)*batch_size,:,:,:]    # real images
#                 noise_gen = MC_input[i*batch_size:(i+1)*batch_size,:,:,:]
#                 generated_images = generator.predict(noise_gen)                     # generated images
#                 #print 'Generator prediction sahape', generated_images.shape
#     #            if i == 0:
#     #                plt.imshow(np.reshape(generated_images[0], (76, 350)), aspect='auto')
#     #                plt.show()
#
#                 #print 'Iteration', str(i+1),'/',iterations_per_epoch
#                 X = np.concatenate((image_batch, generated_images))
#                 y = np.zeros([2*batch_size,2])   # class vector
#                 y[0:batch_size,1] = 1
#                 y[batch_size:,0] = 1
#
#                 # Train the discriminator on the mini-batch
#                 #print 'Train the discriminator on the mini-batch'
#                 d_loss, d_acc  = discriminator.train_on_batch(X,y)
#                 #print 'Discriminator trained'
#                 losses["d"].append(d_loss)
#                 discriminator_acc.append(d_acc)
#
#                 # Create a mini-batch of data (X: noise, y: class vectors pretending that these produce real images)
#                 noise_tr = MC_input[i*batch_size:(i+1)*batch_size,:,:,:]
#     #            for i in range(10):
#     #                plt.imshow(np.reshape(noise_tr[i], (76, 350)), aspect='auto')
#     #                plt.show()
#
#                 y2 = np.zeros([batch_size,2])
#                 y2[:,1] = 1
#                 g_loss = GAN.train_on_batch(noise_tr, y2)
#                 losses["g"].append(g_loss)
#             GAN.save_weights(path_save + "GAN/weights-{epoch:03d}.hdf5")

        # - Plot the loss of discriminator and generator as function of iterations
        #
        # plt.semilogy(losses["d"], label='discriminitive loss')
        # plt.semilogy(losses["g"], label='generative loss')
        # plt.legend()
        # # plt.show()
        # plt.savefig(path_save + 'loss.pdf')
        # plt.close()
        # plt.clf()
        #
        # # - Plot the accuracy of the discriminator as function of iterations
        #
        # plt.plot(discriminator_acc, label='discriminator accuracy')
        # plt.legend()
        # # plt.show()
        # plt.savefig(path_save + 'discriminator_acc.pdf')
        # plt.close()
        # plt.clf()


'''
            counter = 0            
            while counter < 5:
                g_loss = GAN.train_on_batch(noise_tr, y2)
                counter = counter + 1

            losses["g"].append(g_loss)
    
            discriminator.trainable = False
            # Train the generator part of the GAN on the mini-batch
            counter = 0
            if len(discriminator_acc) != 0:
                while discriminator_acc[len(discriminator_acc) - 1] >= 0.7:
                    g_loss = GAN.train_on_batch(noise_tr, y2)
                    counter = counter + 1
                    if counter % 10 == 0:
                        print counter
                        discriminator.trainable = True
                        discriminator.train_on_batch(X,y)
                        discriminator.trainable = False
                        print discriminator_acc[len(discriminator_acc) - 1], len(discriminator_acc)
                    if counter == 100000:
                        print 'Getting out counter max reached'
                        exit()
'''












'''



for i in range (1):
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r") #reads just the file
    column = []
    #print fIN.keys()
    #print fIN.values()[0].shape
    for event in xrange(fIN.values()[0].shape[0]):
        #print fIN.values()[0].shape[0] #result is 8000 the number of lines
        #print fIN['CCPosZ'][event].shape #array that shows the line and what's inside every line (8000,38, 350, 1) or so...
        z_temp = fIN['CCPosZ'][event]
        wfs = fIN['wfs'][event][[0, 2]] #get_wfs(fIN, event) #, index='positive'))
        if all (z>=0. for z in z_temp):
            wfs = wfs[1]
            #column.append(get_wfs(fIN, event, index='positive'))
            #print column.shape
            #plt.imshow(np.reshape(column[column.shape[0]], (38, 350)), aspect='auto')
            #plt.show()
            print 'showing TPC: NEGATIVE', fIN['IsMC'][event]
            plt.imshow(wfs[...,0], aspect='auto')
            plt.show()
        elif all (z<=0. for z in z_temp):
            wfs = wfs[0]
            print 'showing TPC: POSITIVE', fIN['IsMC'][event]
            plt.imshow(wfs[...,0], aspect='auto')
            plt.show()
            #column.append(get_wfs(fIN, event, index='negative'))
            #print column.shape
            #plt.imshow(np.reshape(column[column.shape[0]], (38, 350)), aspect='auto')
            #plt.show()
        else:
            pass
        if event == 10:
            exit()
'''
#add to program
'''

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/mixed_WFs_S5_Th228_P2/"

no_files = len(os.listdir( path )) #this function gives the number of files in the directory


for i in range (len(os.listdir( path ))):  #this function gives the number of files in the directory
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r") #reads just the file
    for event in xrange(fIN.values()[0].shape[0]):
        z_temp = fIN['CCPosZ'][event]
        wfs = fIN['wfs'][event][[0, 2]] #get_wfs(fIN, event) #, index='positive'))
        if all (z>=0. for z in z_temp):
            wfs = wfs[1]
       elif all (z<=0. for z in z_temp):
            wfs = wfs[0]
        else:
            pass
 
'''   
    


'''

    for j in range (20):
        plt.imshow(np.reshape(fIN['wfs'][j][0], (38, 350)), aspect='auto')
        plt.show()
    continue
    for j in range (38):
        plt.plot(fIN['wfs'][0][0][j]+j*20)
        #see note
    plt.show()

note
[wfs] this takes just this column from the data
the [][][]
1st is the number of events (4000) that are in the file
2nd is the wire u1, v1, u2, v2 (4)
3rd is the channel: wires that are used (38)
4th is the value from the amplitude (350)
the j*20 means that the plots are separated by a *20 factor

'''




'''
    line = f.readline()
    line = line.split(" ")
    print line
    index = line.index("wfs") #in a vector tells the position of that element
    lines=f.readlines()
    for x in lines:
        result.append(x.split(' ')[index])
    print index
f.close()
print result
##############################################

#Chose the 
'''
