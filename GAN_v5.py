#how to read files, and use them as input


                                #######################
                                #                     #
                                # Modified Version v5 #
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
from keras.layers import Input, Dense, Reshape, Flatten
from matplotlib import colors
from keras import layers

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
batch_size=40

######################
                    ##
# status = 'test'   ##
status = 'train'  ##
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

#
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
        y = Conv2D(128, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(input_features)
        y = BatchNormalization()(y)
        y = Conv2D(128, (nb_kernel_cols, nb_kernel_rows), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-2))(y)
        y = BatchNormalization()(y)

        y = layers.merge.add([y, input_features])
        # y = layers.Activation('relu')(y)
        return y


    x = Conv2D(64, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-4))(input_image_tensor)
    x = Conv2D(64, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1.e-4))(x)
    x = Conv2D(128, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(1.e-4))(x)
    x = Conv2D(128, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(1.e-4))(x)
    x = Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(1.e-4))(x)


    x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2))(x)

    return x




# def refiner_network(input_image_tensor):
#     """
#     The refiner network, R(theta), is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
#     than holistically modifying the image content, preserving the global structure and annotations.
#     :param input_image_tensor: Input tensor that corresponds to a synthetic image.
#     :return: Output tensor that corresponds to a refined synthetic image.
#     """
#     def resnet_block(input_features, nb_features=32, nb_kernel_rows=3, nb_kernel_cols=5):
#         """
#         A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
#         each with `nb_features` feature maps.
#         See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
#         :param input_features: Input tensor to ResNet block.
#         :return: Output tensor from ResNet block.
#         """
#         y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu',
#                    kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(input_features)
#
#         y = Conv2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same', activation='relu',
#                    kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(y)
#
#         y = layers.merge.add([y, input_features])
#
#         return y
#
#     x = Conv2D(32, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(input_image_tensor)
#
#     x = Conv2D(32, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)
#
#     y = Conv2D(32, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(input_image_tensor)
#
#     x = layers.merge.add([y, x])
#
#     for _ in range(9):
#         x = resnet_block(x, 32)
#
#     x1 = Conv2D(64, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)
#
#     x1 = Conv2D(64, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x1)
#
#     y1 = Conv2D(64, (3, 5), padding='same', activation='relu',
#                kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(1.e-4))(x)
#
#     x = layers.merge.add([y1, x1])
#
#     for _ in range(9):
#         x = resnet_block(x, 64)
#
#     x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2), name='generator_output')(x)
#
#     return x


def critic_inception(input_image_tensor):

    x = Conv2D(64, (3, 5), padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
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

    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)

    x = Dense(2, activation='softmax')(x)

    return x


def make_trainable(model, trainable):
    """ Helper to freeze / unfreeze a model """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

def create_model(model, layer_in, name=''):
    layer_out = model(layer_in)
    return Model(input=layer_in, output=layer_out, name=name)

discriminator = create_model(model=critic_inception, layer_in=layers.Input(shape=(38, 350, 1)), name='critic')
generator = create_model(model=refiner_network, layer_in=layers.Input(shape=(38, 350, 1)), name='refiner')


d_opt = Adam(lr=1e-4, beta_1=0.5, decay=0.0)
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
refined_image_tensor = refiner_network(synthetic_image_tensor)
refined_or_real_image_tensor = layers.Input(shape=(38, 350, 1))
# discriminator_output = discriminator(refined_or_real_image_tensor)
discriminator_output = discriminator(refined_image_tensor)

GAN = Model(input=synthetic_image_tensor, output=discriminator_output, name='refiner')

g_opt = Adam(lr=1e-4, beta_1=0.5, decay=0.0)
make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
GAN.compile(loss='binary_crossentropy', optimizer=g_opt)


print('\nGenerator')
print(generator.summary())

print('\nDiscriminator')
print(discriminator.summary())

# print('\nGenerative Adversarial Network')
# print(GAN.summary())


'''
make trainable freeze part of the model
the GAN is created

'''



# Set up GAN by stacking the discriminator on top of the generator


# Compile saves the trainable status of the model --> After the model is compiled, updating using make_trainable will have no effect


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
    print(generator.summary())

    print('\nDiscriminator')
    print(discriminator.summary())

    # print('\nGenerative Adversarial Network')
    # print(GAN.summary())

    sys.stdout = orig_stdout
    f.close()

save_in_file()



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

            MC_input, y_MC = gen_fake.next()
            X_train, y_train = gen_real.next()

            generated_images = generator.predict(MC_input)  # generated images

            X = np.concatenate((X_train, generated_images))
            y = np.concatenate((y_train, y_MC))  # class vector

            d_loss, d_acc = discriminator.train_on_batch(X, y)
            losses["d"].append(d_loss)
            discriminator_acc.append(d_acc)

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
            if i % (train_steps_per_epoch // 20) == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")

                np.savetxt(path_save + 'discriminator_loss.txt', (losses["d"]), delimiter=',')
                np.savetxt(path_save + 'discriminator_acc.txt', (discriminator_acc), delimiter=',')
                np.savetxt(path_save + 'generator_loss.txt', (losses["g"]), delimiter=',')

                discriminator_acc_mean.append(np.mean(discriminator_acc))
                discriminator_acc = []
                losses_mean["d"].append(np.mean(losses["d"]))
                losses_mean["g"].append(np.mean(losses["g"]))
                losses = {"d": [], "g": []}

                try:

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
                    plt.plot(losses["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.7)
                    plt.plot(losses["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                    plt.title('Loss')
                    plt.legend()
                    plt.xlabel('Training Steps')
                    plt.savefig(path_save + 'loss.pdf')
                    plt.close()
                    plt.clf()

                    plt.semilogy(losses["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.7)
                    plt.semilogy(losses["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                    plt.title('Loss')
                    plt.legend()
                    plt.xlabel('Training Steps')
                    plt.savefig(path_save + 'loss_log.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc, color='green', label='discriminator accuracy', marker='.', linestyle='None', markersize=1)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    #plt.legend()
                    plt.savefig(path_save + 'discriminator_acc_no-line.pdf')
                    plt.close()
                    plt.clf()

                    plt.plot(discriminator_acc, color='green', label='discriminator accuracy', linewidth=0.5)
                    plt.title('Discriminator Accuracy')
                    plt.xlabel('Training Steps')
                    plt.ylabel('Accuracy')
                    #plt.legend()
                    plt.savefig(path_save + 'discriminator_acc.pdf')
                    plt.close()
                    plt.clf()


                except:

                    print 'Plotting loss and accuracy not successful'

        generator.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        discriminator.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        # GAN.save_weights(path_save + "GAN/GAN_weights-" + str(epoch) + ".hdf5")
        print 'Saved weights'



print 'Calling train_for_n_epochs'

generator.save(path_save + "GAN/generator-000" + ".hdf5")
discriminator.save(path_save + "GAN/discriminator-000" + ".hdf5")
# GAN.save(path_save + "GAN/GAN-000.hdf5")

generator.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
discriminator.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")
# GAN.save_weights(path_save + "GAN/GAN_weights-000.hdf5")

train_for_n(epochs=100, batch_size=batch_size)

