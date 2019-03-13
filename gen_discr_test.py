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
path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_03_07-16_46_20/GAN/'
#path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_01_24-11_45_21/GAN/'
path_discriminator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_03_06-17_07_47/GAN/'
path_save = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/gen_discr_test/'

batch_size = 30

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
    generator.load_weights(path_generator + 'generator_weights-43.hdf5')
    print '\nGenerator'
    generator.summary()
    return generator

def load_discriminator():
    import keras as ks
    discriminator = ks.models.load_model(path_discriminator + 'discriminator-000.hdf5')
    #discriminator.load_weights(path_discriminator + 'discriminator_weights-89.hdf5')
    print '\nDiscriminator'
    discriminator.summary()
    return discriminator

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
    # model.add(Conv2D(32, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
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
    #model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation='softmax'))
    return model






generator = load_generator()
# discriminator = load_discriminator()
discriminator = build_discriminator()
d_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
print '\nDiscriminator'
discriminator.summary()

# generated_images = generator.predict(MC_input)
# X = np.concatenate((X_train, generated_images))
# X = np.split(X[:], 2, axis=1)
# X = np.swapaxes(X, 2, 3)
# X = list(X)
# y = np.concatenate((y_train, y_MC))

#discriminator.fit(X,y, epochs=1, batch_size=32)



def train_for_n(epochs=50, batch_size=batch_size):
    discriminator_loss = []
    generator_loss = []
    epoch_counter = []
    epoch_counter_fine = []
    losses = {"d": [], "t": []}
    discriminator_acc = {"d": [], "t": []}
    losses_mean = {"d": [], "t": []}
    discriminator_acc_mean = {"d": [], "t": []}

    for epoch in range(epochs):
        print 'epoch', str(epoch + 1), '/', epochs

        train_steps_per_epoch = int(getNumEvents(files) / batch_size / 2.)

        for i in range(train_steps_per_epoch):
            # if i % 100 == 0:
            #     print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")
                # if i != 0:
                #     print 'Discriminator accuracy', discriminator_acc[len(discriminator_acc) - 1]
            MC_input, y_MC = gen_fake.next()
            X_train, y_train = gen_real.next()

            generated_images = generator.predict(MC_input)
            '''
            #generated_images = MC_input
            X = np.concatenate((X_train, generated_images))
            # X = np.concatenate((X, X), axis=1)
            # print X.shape
            # exit()
            shape = (2, batch_size, 38, 350, 1)
            X_temp = np.zeros(shape)
            X_temp[0, :, :, :, :] = X
            X_temp[1, :, :, :, :] = X
            # X = np.split(X[:], 2, axis=1)
            X = np.swapaxes(X_temp, 2, 3)
            '''
            X = np.concatenate((X_train, generated_images))#list(X)
            y = np.concatenate((y_train, y_MC))

            d_loss, d_acc = discriminator.train_on_batch(X, y)
            # print 'Discriminator trained'
            losses["d"].append(d_loss)
            discriminator_acc["d"].append(d_acc)



            MC_input_test, y_MC_test = gen_fake_test.next()
            X_test, y_test = gen_real_test.next()

            generated_images_test = generator.predict(MC_input_test)
            '''
            #generated_images_test = MC_input_test
            Xt = np.concatenate((X_test, generated_images_test))
            Xt_temp = np.zeros(shape)
            Xt_temp[1, :, :, :, :] = Xt
            Xt_temp[0, :, :, :, :] = Xt
            # Xt = np.split(Xt_temp[:], 0, axis=0)
            Xt = np.swapaxes(Xt_temp, 2, 3)
            '''
            Xt = np.concatenate((X_test, generated_images_test))#list(Xt)
            yt = np.concatenate((y_test, y_MC_test))

            d_loss_test, d_acc_test = discriminator.evaluate(Xt, yt, verbose=0)
            losses["t"].append(d_loss_test)
            discriminator_acc["t"].append(d_acc_test)

            # if i % 100 == 0:
            if i % (train_steps_per_epoch // 20) == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")

                discriminator_acc_mean["d"].append(np.mean(discriminator_acc["d"]))
                discriminator_acc_mean["t"].append(np.mean(discriminator_acc["t"]))
                discriminator_acc = {"d": [], "t": []}
                losses_mean["d"].append(np.mean(losses["d"]))
                losses_mean["t"].append(np.mean(losses["t"]))
                losses = {"d": [], "t": []}

                plt.plot(losses_mean["d"], label='Training loss')#, linewidth=0.5, alpha=0.7)
                plt.plot(losses_mean["t"], label='Test loss')#, linewidth=0.5, alpha=0.5)
                plt.title('Discriminitive Loss')
                plt.legend()
                plt.xlabel('Training Steps')
                plt.savefig(path_save + 'loss.pdf')
                plt.close()
                plt.clf()

                plt.semilogy(losses_mean["d"], label='Training loss')#, linewidth=0.5, alpha=0.7)
                plt.semilogy(losses_mean["t"], label='Test loss')#, linewidth=0.5, alpha=0.5)
                plt.title('Discriminitive Loss')
                plt.legend()
                plt.xlabel('Training Steps')
                plt.savefig(path_save + 'loss_log.pdf')
                plt.close()
                plt.clf()

                plt.plot(discriminator_acc_mean["d"], label='Train accuracy')#, linewidth=0.5, alpha=0.7)
                plt.plot(discriminator_acc_mean["t"], label='Test accuracy')#, linewidth=0.5, alpha=0.5)
                plt.title('Discriminator Accuracy')
                plt.legend()
                plt.xlabel('Training Steps')
                plt.ylabel('Accuracy')
                # plt.legend()
                plt.savefig(path_save + 'discriminator_acc.pdf')
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
                    plt.plot(np.subtract(generated_images[0][j], MC_input[0][j]) + j * 20, color='black', linewidth=0.5)
                    plt.ylabel('Mod - Orig')
                    plt.xlabel('Time steps')
                    plt.yticks([])
                    plt.xlim(0, 350)
                plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot_diff.pdf', bbox_inches='tight')
                plt.close()
                plt.clf()

        discriminator.save_weights(path_save + "model/discriminator_weights-" + str(epoch) + ".hdf5")
        try:
            np.savetxt(path_save + 'discriminator_loss_train.txt', (losses_mean["d"]), delimiter=',')
            np.savetxt(path_save + 'discriminator_acc_train.txt', (discriminator_acc_mean["d"]), delimiter=',')
            np.savetxt(path_save + 'discriminator_loss_test.txt', (losses_mean["t"]), delimiter=',')
            np.savetxt(path_save + 'discriminator_acc_test.txt', (discriminator_acc_mean["t"]), delimiter=',')
        except:
            print 'Not able to save loss and accuracy'

train_for_n(epochs=50, batch_size=batch_size*2)




exit()
for i in range(20):

    print 'Get events'

    MC_input, y_MC = gen_fake.next()
    #X_train, y_train = gen_real.next()

    print 'Predicting images'

    generated_images = generator.predict(MC_input)

    if np.max(np.abs(generated_images[0])) > np.max(np.abs(MC_input[0])):
        fmax = np.max(np.abs(generated_images[0]))
    else:
        fmax = np.max(np.abs(MC_input[0]))

    plt.subplot(2, 1, 1)
    plt.imshow(np.reshape(generated_images[0], (76, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'),
               norm=colors.Normalize(vmin=-fmax, vmax=fmax))
    plt.subplot(2, 1, 2)
    plt.imshow(np.reshape(MC_input[0]*10., (76, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'),
               norm=colors.Normalize(vmin=-fmax, vmax=fmax))
    plt.show()
    # plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + '.pdf', bbox_inches='tight')
    # plt.close()
    # plt.clf()