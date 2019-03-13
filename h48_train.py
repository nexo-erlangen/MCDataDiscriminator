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
path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/GAN/'
#path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_01_24-11_45_21/GAN/'
path_discriminator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/GAN/'
path_save = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/'
path_model = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/GAN/'


last_epoch = 43



batch_size = 32

result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory
files = []
for i in range(15):
    files.append(path + str(i) + "-shuffled.hdf5")
fake = {'IsMC': [0]}
real = {'IsMC': [1]}

gen_fake = generate_batches_from_files(files, batch_size, wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)
gen_real = generate_batches_from_files(files, batch_size , wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=real, yield_mc_info=0)

files_test = []
for i in range(15, no_files):
    files_test.append(path + str(i) + "-shuffled.hdf5")
gen_fake_test = generate_batches_from_files(files_test, batch_size, wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)
gen_real_test = generate_batches_from_files(files_test, batch_size , wires='V', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=real, yield_mc_info=0)

# MC_input, y_MC = gen_fake.next()
# X_train, y_train = gen_real.next()


def load_generator():
    import keras as ks
    generator = ks.models.load_model(path_generator + 'generator-000.hdf5')
    # generator.load_weights(path_generator + 'generator_weights-' + str(last_epoch) + '.hdf5')
    return generator

def load_discriminator():
    import keras as ks
    discriminator = ks.models.load_model(path_discriminator + 'discriminator-000.hdf5')
    # discriminator.load_weights(path_discriminator + 'discriminator_weights-' + str(last_epoch) + '.hdf5')
    return discriminator

def make_trainable(model, trainable):
    """ Helper to freeze / unfreeze a model """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

#generator = load_generator()
# discriminator = load_discriminator()

def refiner_network(input_image_tensor):
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
    x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
    return x

def build_discriminator(drop_rate=0.25):
    model = Sequential()
    model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(64, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(128, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))
    return model
#
# discriminator = build_discriminator()
#
#
# d_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
# discriminator.load_weights(path_discriminator + 'discriminator_weights-' + str(last_epoch) + '.hdf5')
# discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
#
#
# synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
# refined_image_tensor = refiner_network(synthetic_image_tensor)
# refined_or_real_image_tensor = layers.Input(shape=(38, 350, 1))
# # discriminator_output = discriminator(refined_or_real_image_tensor)
# discriminator_output = discriminator(refined_image_tensor)
#
# generator = Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner_gen')
# generator.load_weights(path_generator + 'generator_weights-' + str(last_epoch) + '.hdf5')
# # generator_model = Model(input=generator_model_test.input, output=discriminator_output, name='refiner_dis')
# GAN = Model(input=synthetic_image_tensor, output=discriminator_output, name='refiner')
# # GAN.load_weights(path_generator + 'GAN_weights-' + str(last_epoch) + '.hdf5')
# g_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
# make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
# GAN.compile(loss='binary_crossentropy', optimizer=g_opt)



discriminator = build_discriminator()

d_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

synthetic_image_tensor = layers.Input(shape=(38, 350, 1))
refined_image_tensor = refiner_network(synthetic_image_tensor)
refined_or_real_image_tensor = layers.Input(shape=(38, 350, 1))
# discriminator_output = discriminator(refined_or_real_image_tensor)
discriminator_output = discriminator(refined_image_tensor)

generator = Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner_gen')
# generator_model = Model(input=generator_model_test.input, output=discriminator_output, name='refiner_dis')
GAN = Model(input=synthetic_image_tensor, output=discriminator_output, name='refiner')

g_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
GAN.compile(loss='binary_crossentropy', optimizer=g_opt)

discriminator.load_weights(path_discriminator + 'discriminator_weights-' + str(last_epoch) + '.hdf5')
generator.load_weights(path_generator + 'generator_weights-' + str(last_epoch) + '.hdf5')
#GAN.load_weights(path_generator + 'GAN_weights-' + str(last_epoch) + '.hdf5')




#Print summary
print '\nGenerator'
generator.summary()
print '\nDiscriminator'
discriminator.summary()
print '\nGAN'
GAN.summary()

#GAN.load_weights(path_generator + 'GAN_weights-' + str(last_epoch) + '.hdf5')
#discriminator.load_weights(path_discriminator + 'discriminator_weights-' + str(last_epoch) + '.hdf5')


# generated_images = generator.predict(MC_input)
# X = np.concatenate((X_train, generated_images))
# X = np.split(X[:], 2, axis=1)
# X = np.swapaxes(X, 2, 3)
# X = list(X)
# y = np.concatenate((y_train, y_MC))

#discriminator.fit(X,y, epochs=1, batch_size=32)

# loss vector


def train_for_n(epochs=50, batch_size=32):
    discriminator_loss = []
    generator_loss = []
    discriminator_loss_average = []
    generator_loss_average = []
    epoch_counter = []
    epoch_counter_fine = []

    losses = {"d": [], "g": []}
    discriminator_acc = []
    losses_mean = {"d": [], "g": []}
    discriminator_acc_mean = []

    for epoch in range(last_epoch + 1, epochs):
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
            '''
            acc = []
            loss = []
            for k in range(10):
                print k+1, '/ 10'
                MC_input, y_MC = gen_fake.next()
                X_train, y_train = gen_real.next()
                # print xs_i.shape
                # print ys_i.shape
                # noise_mask = ys_i[:, 1] == 1
                # real_mask = np.invert(noise_mask)
                # noise_gen = xs_i[noise_mask]
                # real_gen = xs_i[real_mask]
                generated_images = generator.predict(MC_input)  # generated images
                X = np.concatenate((X_train, generated_images))
                y = np.concatenate((y_train, y_MC))
                pretrain_loss, pretrain_acc = discriminator.evaluate(X, y, verbose=0, batch_size=32)
                acc.append(pretrain_acc)
                loss.append(pretrain_loss)
            plt.subplot(2, 1, 1)
            plt.plot(acc, color='blue', linewidth=0.5)
            plt.ylabel('Accuracy')
            plt.xticks([])
            plt.subplot(2, 1, 2)
            plt.plot(loss, color='orange', linewidth=0.5)
            plt.ylabel('Loss')
            plt.show()

            exit()


            for a in range(generated_images.shape[0]):
                for j in range(MC_input.shape[1]):
                    plt.subplot(3, 1, 1)
                    plt.plot(generated_images[a][j] + j * 20, color='black', linewidth=0.5)
                    # plt.title('[Top] Modified MC, [Center] MC, [Bottom] ModMC-MC')
                    plt.ylabel('Modified MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(3, 1, 2)
                    plt.plot(MC_input[a][j] + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Original MC')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0, 350)
                    plt.subplot(3, 1, 3)
                    plt.plot(np.subtract(generated_images[a][j], MC_input[a][j]) + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Mod - Orig')
                    plt.xlabel('Time steps')
                    plt.yticks([])
                    plt.xlim(0, 350)
                plt.show()

            exit()
            '''
            MC_input, y_MC = gen_fake.next()
            X_train, y_train = gen_real.next()
            # print 'Generator prediction sahape', generated_images.shape
            #            if i == 0:
            #                plt.imshow(np.reshape(generated_images[0], (76, 350)), aspect='auto')
            #                plt.show()

            # print 'Iteration', str(i+1),'/',iterations_per_epoch

            generated_images = generator.predict(MC_input)  # generated images

            X = np.concatenate((X_train, generated_images))
            y = np.concatenate((y_train, y_MC))  # class vector
            # y[0:xs_i[real_mask].shape[0], 1] = 1
            # y[generated_images.shape[0]:, 0] = 1

            d_loss, d_acc = discriminator.train_on_batch(X, y)
            losses["d"].append(d_loss)
            discriminator_acc.append(d_acc)

            g_loss = GAN.train_on_batch(MC_input, y_train) #original y_MC
            losses["g"].append(g_loss)
            '''
            if len(discriminator_acc) >= train_steps_per_epoch:
                discriminator_acc_mean.append(np.mean(discriminator_acc))
                discriminator_acc = []
                losses_mean["d"].append(np.mean(losses["d"]))
                losses_mean["g"].append(np.mean(losses["g"]))
                losses = {"d": [], "g": []}
            '''

            if i % 100 == 0:
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
                        plt.title('[Top] Modified MC, [Bottom] MC')
                        plt.subplot(2, 1, 2)
                        plt.plot(MC_input[0][j] + j * 20, color='black', linewidth=0.5)
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

                plt.semilogy(losses["d"], label='Discriminitive loss', linewidth=0.5, alpha=0.5)
                plt.semilogy(losses["g"], label='Generative loss', linewidth=0.5, alpha=0.5)
                plt.title('Loss')
                plt.legend()
                plt.xlabel('Training Steps')
                plt.savefig(path_save + 'loss.pdf')
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


                # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                np.savetxt(path_save + 'discriminator_loss.txt', (losses["d"]), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                np.savetxt(path_save + 'generator_loss.txt', (losses["g"]), delimiter=',')
                np.savetxt(path_save + 'discriminator_acc.txt', (discriminator_acc), delimiter=',')

        generator.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        discriminator.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        GAN.save_weights(path_save + "GAN/GAN_weights-" + str(epoch) + ".hdf5")
        print 'Saved weights'

print 'Calling Traing'
train_for_n(epochs=500, batch_size=batch_size)#*2)


#
#
# exit()
# for i in range(20):
#
#     print 'Get events'
#
#     MC_input, y_MC = gen_fake.next()
#     #X_train, y_train = gen_real.next()
#
#     print 'Predicting images'
#
#     generated_images = generator.predict(MC_input)
#
#     if np.max(np.abs(generated_images[0])) > np.max(np.abs(MC_input[0])):
#         fmax = np.max(np.abs(generated_images[0]))
#     else:
#         fmax = np.max(np.abs(MC_input[0]))
#
#     plt.subplot(2, 1, 1)
#     plt.imshow(np.reshape(generated_images[0], (76, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'),
#                norm=colors.Normalize(vmin=-fmax, vmax=fmax))
#     plt.subplot(2, 1, 2)
#     plt.imshow(np.reshape(MC_input[0]*10., (76, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'),
#                norm=colors.Normalize(vmin=-fmax, vmax=fmax))
#     plt.show()
#     # plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + '.pdf', bbox_inches='tight')
#     # plt.close()
#     # plt.clf()