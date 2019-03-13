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
from keras.layers import Input, Dense, Reshape, Flatten
from matplotlib import colors
from keras import layers

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
batch_size=32

######################
                    ##
status = 'train'    ##
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
# def one_epoch():
#     MC_input = []
#     DATA_input = []
#
#     for i in range(1):#no_files):
#         fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
#         for event in xrange(fIN.values()[0].shape[0]):
#             if fIN['IsMC'][event] == 1.0:
#                 MC_input.append(fIN['wfs'][event])
#             elif fIN['IsMC'][event] == 0.0:
#                 DATA_input.append(fIN['wfs'][event])
#             else:
#                 print 'There is a problem with the reading data'
#         print 'Data from file ', str(i+1), '/', str(no_files), ' acquired.'
#     MC_input = np.asarray(MC_input)
#     DATA_input = np.asarray(DATA_input)
#     wireindex = [1, 3]
#
#     MC_input = MC_input[:,wireindex,:,:,:]
#     DATA_input = DATA_input[:,wireindex,:,:,:]
#     MC_temp = []
#     DATA_temp = []
#     print 'Concatenating'
#
#     for i in range(MC_input.shape[0]):
#         MC_temp.append(np.concatenate((MC_input[i][0],MC_input[i][1]),axis=0))
#         DATA_temp.append(np.concatenate((DATA_input[i][0],DATA_input[i][1]),axis=0))
#     MC_input = np.array(MC_temp)
#     DATA_input = np.array(DATA_temp)
#     #np.reshape(MC_input, (MC_input.shape[0], 76, -1))
#     #print MC_input.shape
#     frac_test = int(DATA_input.shape[0]*0.1)
#     X_test = DATA_input[:frac_test]
#     X_train = DATA_input[frac_test:]
#     return MC_input, X_test, X_train

MC_input, y_MC = gen_fake.next()
X_train, y_train = gen_real.next()
# def graf(image):
#     plt.imshow(np.reshape(image[0], (76, 350)), aspect='auto')
#     plt.show()
# graf(MC_input)
# graf(X_train)
#Missing
#X_test
'''
start the generator building

'''
#in Conv2D activation = LeakyReLU(0.2), is used
# def build_generator():
#     model = Sequential(name='generator')
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
#     model.add(Conv2D(1, (1, 1), padding='same'))
#     return model


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

    # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
    # x_1 = layers.Convolution2D(128, 1, 1, border_mode='same', activation='relu')(input_image_tensor)
    # x_3 = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu')(input_image_tensor)
    # x_5 = layers.Convolution2D(128, 5, 5, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    # for _ in range(3):
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)

    # x = layers.Convolution2D(128, 3, 3, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 5, 5, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)


    # x = layers.Convolution2D(128, 7, 1, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(input_image_tensor)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 1, 7, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 9, 1, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 1, 9, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 11, 1, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 1, 11, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 13, 1, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)
    # x = layers.Convolution2D(128, 1, 13, border_mode='same', activation='relu',
    #                          kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = BatchNormalization()(x)


    x = resnet_block(input_image_tensor, nb_features=128, nb_kernel_rows=7, nb_kernel_cols=1)
    x = resnet_block(x, nb_features=128, nb_kernel_rows=9, nb_kernel_cols=1)
    x = resnet_block(x, nb_features=128, nb_kernel_rows=11, nb_kernel_cols=1)
    x = resnet_block(x, nb_features=128, nb_kernel_rows=13, nb_kernel_cols=1)
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
    x = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1.e-2))(x)
    # x = layers.merge.add([x, input_image_tensor])

    return x

'''
def build_generator():
    generator = Sequential(name='generator')

    generator.add(Conv2DTranspose(256, 4,  use_bias=False, strides=2,
                                        padding="SAME", activation=None, input_shape=(76, 350, 1)))
    generator.add(BatchNormalization(scale=False,  name="bn1"))

    generator.add(Conv2DTranspose(128, 4,  use_bias=False, strides=2,
                                        padding="SAME", activation=None))
    generator.add(BatchNormalization(scale=False, name="bn2"))

    generator.add(Conv2DTranspose(64, 4, use_bias=False, strides=2,
                                        padding="SAME", activation=None))
    generator.add(BatchNormalization(scale=False, name="bn3"))

    # pass the number of filters of the current feature volume
    #generator.add(SelfAttention(64))

    generator.add(Conv2DTranspose(32, 4,   use_bias=False, strides=2,
                                        padding="SAME", activation=None))
    generator.add(BatchNormalization(scale=False, name="bn4"))

    generator.add(Conv2DTranspose(16, 4, use_bias=False, strides=2,
                                        padding="SAME", activation=None))
    generator.add(BatchNormalization(scale=False,   name="bn5"))

    generator.add(Conv2D(3, 3, strides=1,  padding='SAME', activation=None))
    #generator.add(Activation(activation='tanh'))

'''
'''
#in Conv2D activation = LeakyReLU(0.2) is used
def build_generator():
    generator = Sequential(name='generator')
    #generator.add(Dense(7 * 7 * 128, input_shape=(latent_dim,76,350,)))
    generator.add(Conv2D(32, (5, 5), padding='same', input_shape=(76, 350, 1)))
    generator.add(BatchNormalization())
    # generator.add(Activation('relu'))
    generator.add(Conv2D(64, (5, 5), padding='same'))
    generator.add(BatchNormalization())
    # generator.add(Activation('relu'))
    #generator.add(UpSampling2D(size=(2,2)))
    generator.add(Conv2D(128, (5, 5), padding='same'))
    generator.add(BatchNormalization())
    # generator.add(Activation('relu'))
    #generator.add(UpSampling2D(size=(2,2)))
    generator.add(Conv2D(256, (5, 5), padding='same'))
    generator.add(BatchNormalization())
    # generator.add(Activation('relu'))
    generator.add(Conv2D(1, (5, 5), padding='same')) #, activation='relu'))
    return generator
'''

#set up a generator


'''
start the discriminator building

'''

# def build_discriminator(drop_rate=0.25):
#     """ Discriminator network """
#     discriminator = Sequential(name='discriminator')
#     discriminator.add(Conv2D(64, (3, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(38, 350, 1)))
#     discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(drop_rate))
#     discriminator.add(Conv2D(128, (3, 5), padding='same', strides=(2, 2), activation='relu'))
#     discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(drop_rate))
#     discriminator.add(Conv2D(256, (3, 5), padding='same', strides=(2, 2), activation='relu'))
#     discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(drop_rate))
#     discriminator.add(Flatten())
#     #discriminator.add(Dense(128, activity_regularizer=l1_l2(1e-5)))
#     #discriminator.add(LeakyReLU(0.2))
#     discriminator.add(Dropout(drop_rate))
#     discriminator.add(Dense(2, activation='softmax'))
#     return discriminator


def build_discriminator(drop_rate=0.25):
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
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





# def build_discriminator(drop_rate=0.25):
#     model = Sequential(name='discriminator')
#     model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
#     model.add(MaxPooling2D(pool_size=(2, 4)))
#
#     model.add(Conv2D(64, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2)))
#     model.add(MaxPooling2D(pool_size=(2, 4)))
#
#     model.add(Conv2D(128, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2)))
#     model.add(MaxPooling2D(pool_size=(2, 4)))
#
#     model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(1.e-2)))
#     model.add(MaxPooling2D(pool_size=(2, 4)))
#
#     model.add(Dropout(drop_rate))
#     model.add(Dense(2, activation='softmax'))
#     return model

def make_trainable(model, trainable):
    """ Helper to freeze / unfreeze a model """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


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



# gan_input = Input(shape=[38, 350, 1])
# gan_output = discriminator(generator(gan_input))
# GAN = Model(gan_input, gan_output)

print('\nGenerator')
print(generator.summary())

print('\nDiscriminator')
print(discriminator.summary())

print('\nGenerative Adversarial Network')
print(GAN.summary())


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

    print('\nGenerative Adversarial Network')
    print(GAN.summary())

    sys.stdout = orig_stdout
    f.close()

save_in_file()


# --------------------------------------------------
# Pretrain the discriminator:
# --------------------------------------------------

# - Create a dataset of 1000 real train images and 1000 fake images.
#ntrain = 100
# no_MC = np.random.choice(MC_input.shape[0], ntrain, replace='False')
# no_train = np.random.choice(X_train.shape[0], ntrain, replace='False')
# real_train = X_train[no_train,:,:,:]   # sample real images from training set
# noise_gen = MC_input[no_MC,:,:,:]   # sample MC images from input(or maybe not) \_(-.-)_/


'''
generated_images = generator.predict(MC_input)

X = np.concatenate((X_train, generated_images))
y = np.concatenate((y_MC, y_train))
#IMPORTANT -> The class was the opposite for the first training
# - Train the discriminator for 1 epoch on this dataset.
discriminator.fit(X, y, epochs=1, batch_size=32)
# - Evaluate the test accuracy of your network.
pretrain_loss, pretrain_acc = discriminator.evaluate(X, y, verbose=0, batch_size=32)
print('Test accuracy: %04f' % pretrain_acc)
# loss vector
losses = {"d":[], "g":[]}
discriminator_acc = []
'''



#TODO Create dataset
# - Create a dataset of 500 real test images and 500 fake images.
# no_MC = np.random.choice(MC_input.shape[0], ntrain//2, replace='False')
# no_test = np.random.choice(X_test.shape[0], ntrain//2, replace='False')
# real_test = X_test[no_test,:,:,:]   # sample real images from test set
# noise_gen = MC_input[no_MC,:,:,:]
#generated_images = generator.predict(MC_input)    # generate fake images with untrained generator



# print 'Generator prediction sahape', generated_images.shape
# Xt = np.concatenate((real_test, generated_images))
# yt = np.zeros([ntrain, 2])   # class vector: one-hot encoding
# yt[:ntrain//2, 1] = 1         # class 1 for real images
# yt[ntrain//2:, 0] = 1         # class 0 for generated images


'''
def crate_batches(j):
    MC_input2 = []
    DATA_input2 = []
    fIN = h5py.File(path + str(j) + "-shuffled.hdf5", "r")
    for event in xrange(fIN.values()[0].shape[0]):
        if fIN['IsMC'][event] == 1.0:
            MC_input2.append(fIN['wfs'][event])
        elif fIN['IsMC'][event] == 0.0:
            DATA_input2.append(fIN['wfs'][event])
        else:
            print 'There is a problem with the reading data'
    #print 'Data from file ', str(i + 1), '/', str(no_files), ' acquired.'
    MC_input2 = np.asarray(MC_input2)
    DATA_input2 = np.asarray(DATA_input2)
    wireindex = [1, 3]

    MC_input2 = MC_input2[:, wireindex, :, :, :]
    DATA_input2 = DATA_input2[:, wireindex, :, :, :]
    MC_temp = []
    DATA_temp = []
    print 'Concatenating'

    for i in range(MC_input2.shape[0]): #Puts together noise and signal
        MC_temp.append(np.concatenate((MC_input2[i][0], MC_input2[i][1]), axis=0))
        DATA_temp.append(np.concatenate((DATA_input2[i][0], DATA_input2[i][1]), axis=0))
    MC_input2 = np.array(MC_temp)
    DATA_input2 = np.array(DATA_temp)
    np.random.shuffle(MC_input2)
    np.random.shuffle(DATA_input2)
    # np.reshape(MC_input, (MC_input.shape[0], 76, -1))
    # print MC_input.shape
    frac_test = int(DATA_input2.shape[0] * 0.1)
    X_test = DATA_input2[:frac_test]
    X_train = DATA_input2[frac_test:]
    return MC_input, X_test, X_train
'''
########################################

#
# for i in range(no_files):
#     fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
#     globals()['MC_input_%s' % i] = []
#     globals()['DATA_input_%s' % i] = []
#     for event in xrange(fIN.values()[0].shape[0]):
#         if fIN['IsMC'][event] == 1.0:
#             globals()['MC_input_%s' % i].append(fIN['wfs'][event])
#         elif fIN['IsMC'][event] == 0.0:
#             globals()['DATA_input_%s' % i].append(fIN['wfs'][event])
#         else:
#             print 'There is a problem with the reading data'
#     print 'Data from file ', str(i + 1), '/', str(no_files), ' acquired.'
'''
def create_batch(j):

    for i in range(3*j, 3*(j+1)):
        fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
        MC_input_all = []
        DATA_input_all = []
        for event in xrange(fIN.values()[0].shape[0]):
            if fIN['IsMC'][event] == 1.0:
                MC_input_all.append(fIN['wfs'][event])
            elif fIN['IsMC'][event] == 0.0:
                DATA_input_all.append(fIN['wfs'][event])
            else:
                print 'There is a problem with the reading data'
        print 'Data from file ', str(i + 1), '/', str(no_files), ' acquired.'

    MC_input = np.asarray(MC_input_all)
    DATA_input = np.asarray(DATA_input_all)
    wireindex = [1, 3]

    MC_input = MC_input[:, wireindex, :, :, :]
    DATA_input = DATA_input[:, wireindex, :, :, :]
    MC_temp = []
    DATA_temp = []
    print 'Concatenating'

    for i in range(MC_input.shape[0]):
        MC_temp.append(np.concatenate((MC_input[i][0], MC_input[i][1]), axis=0))
        DATA_temp.append(np.concatenate((DATA_input[i][0], DATA_input[i][1]), axis=0))
    MC_input = np.array(MC_temp)
    DATA_input = np.array(DATA_temp)
    np.random.shuffle(MC_input)
    np.random.shuffle(DATA_input)
    # np.reshape(MC_input, (MC_input.shape[0], 76, -1))
    # print MC_input.shape
    frac_test = int(DATA_input.shape[0] * 0.1)
    X_test = DATA_input[:frac_test]
    X_train = DATA_input[frac_test:]
    return MC_input, X_test, X_train

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
            # if i % 100 == 0:
            #     print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")
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
            if i % (train_steps_per_epoch // 20) == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch, 'Time', time.strftime("%H:%M")

                discriminator_acc_mean.append(np.mean(discriminator_acc))
                discriminator_acc = []
                losses_mean["d"].append(np.mean(losses["d"]))
                losses_mean["g"].append(np.mean(losses["g"]))
                losses = {"d": [], "g": []}

                try:
                    # if np.max(np.abs(generated_images[0])) > np.max(np.abs(MC_input[0])):
                    #     fmax = np.max(np.abs(generated_images[0]))
                    # else:
                    #     fmax = np.max(np.abs(MC_input[0]))
                    #
                    # plt.subplot(2, 1, 1)
                    # plt.imshow(np.reshape(generated_images[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    # plt.title('[Top] Modified MC, [Bottom] MC')
                    # plt.subplot(2, 1, 2)
                    # plt.imshow(np.reshape(MC_input[0], (38, 350)), aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=colors.Normalize(vmin=-fmax, vmax=fmax))
                    # plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + '.pdf', bbox_inches='tight')
                    # plt.close()
                    # plt.clf()
                    #
                    # for j in range(MC_input.shape[1]):
                    #     plt.subplot(2, 1, 1)
                    #     plt.plot(generated_images[0][j] + j * 20, color='black', linewidth=0.5)
                    #     #plt.title('[Top] Modified MC, [Bottom] MC')
                    #     plt.ylabel('Modified MC')
                    #     plt.xticks([])
                    #     plt.yticks([])
                    #     plt.xlim(0, 350)
                    #     plt.subplot(2, 1, 2)
                    #     plt.plot(MC_input[0][j] + j * 20, color='black', linewidth=0.5)
                    #     plt.ylabel('Original MC')
                    #     plt.xlabel('Time steps')
                    #     plt.yticks([])
                    #     plt.xlim(0, 350)
                    # plt.savefig(path_save + 'wfs_' + str(epoch) + '_' + str(i) + 'plot.pdf', bbox_inches='tight')
                    # plt.close()
                    # plt.clf()

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


print 'Calling train_for_n_epochs'

generator.save(path_save + "GAN/generator-000" + ".hdf5")
discriminator.save(path_save + "GAN/discriminator-000" + ".hdf5")
GAN.save(path_save + "GAN/GAN-000.hdf5")

generator.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
discriminator.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")
GAN.save_weights(path_save + "GAN/GAN_weights-000.hdf5")

train_for_n(epochs=100, batch_size=batch_size)










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
