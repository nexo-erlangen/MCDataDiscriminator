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

                                            #######################
                                            #                     #
                                            # Modified Version v3 #
                                            #                     #
                                            #######################



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
from keras.layers import ReLU, GlobalMaxPooling2D, Dropout, Activation, MaxPooling2D
from keras.activations import linear
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from matplotlib import colors
from keras import regularizers



BATCH_SIZE = 50
batch_size = BATCH_SIZE
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # 10 in the paper

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
status = 'train'


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

#
# def make_generator():
#     """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
#     of size 28x28x1."""
#     model = Sequential()
#     # model.add(Dense(1024, input_dim=100))
#     # model.add(LeakyReLU())
#     # model.add(Dense(128 * 7 * 7))
#     # model.add(BatchNormalization())
#     # model.add(LeakyReLU())
#     # if K.image_data_format() == 'channels_first':
#     #     model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
#     #     bn_axis = 1
#     # else:
#     #     model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
#     #     bn_axis = -1
#     '''strides=2,''' # doubles the dimention of the input
#     model.add(Conv2DTranspose(128, (5, 5), padding='same', input_shape=(76, 350, 1)))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Convolution2D(64, (5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Conv2DTranspose(64, (5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     # Because we normalized training inputs to lie in the range [-1, 1],
#     # the tanh function should be used for the output of the generator to ensure its output
#     # also lies in this range.
#     model.add(Convolution2D(1, (5, 5), padding='same'))
#     return model

def make_generator():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2), input_shape=(38, 350, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(1.e-2)))
    # model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(BatchNormalization())
    model.add(Conv2D(1, (1, 1), padding='same'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    model = Sequential()
    model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2),  input_shape=(38, 350, 1)))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(64, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(128, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(256, (3, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(1.e-2)))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Dense(1,  kernel_initializer='glorot_uniform'))
    return model


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(10, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

#
# parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
# parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated files to")
# args = parser.parse_args()


'''
# First we load the image data, reshape it and normalize it to the range [-1, 1]
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.concatenate((X_train, X_test), axis=0)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
else:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
'''


# Now we initialize the generator and discriminator.
print('\nGenerator')
generator = make_generator()
print(generator.summary())

print('\nDiscriminator')
discriminator = make_discriminator()
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
    print(generator.summary())

    print('\nDiscriminator')
    print(discriminator.summary())

    print 'Training ration:', TRAINING_RATIO   # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
    print 'Gradient penality:', GRADIENT_PENALTY_WEIGHT   # 10 in the paper

    sys.stdout = orig_stdout
    f.close()

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=[38, 350, 1])
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
# print discriminator_layers_for_generator
# exit()
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=(38, 350, 1))
generator_input_for_discriminator = Input(shape=(38, 350, 1))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
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
            if i % 100 == 0:
                print 'Step per epoch', i, '/', train_steps_per_epoch
                # if i != 0:
                #     print 'Discriminator accuracy', discriminator_acc[len(discriminator_acc) - 1]
            MC_input, y_MC = gen_fake.next()
            X_train, y_train = gen_real.next()
            # np.random.shuffle(X_train)
            # np.random.shuffle(MC_input)
            # print("Epoch: ", epoch)
            # print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
            for j in range(TRAINING_RATIO):
                image_batch = X_train
                # print 'before predict'
                noise = generator.predict(MC_input) #original only MC_input
                # print 'after predict'
                discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                             [positive_y, negative_y, dummy_y]))
                epoch_counter_fine.append(epoch+float(i+(float(j)/TRAINING_RATIO))/train_steps_per_epoch)
            generator_loss.append(generator_model.train_on_batch(MC_input, positive_y)) # original: positive_y
            epoch_counter.append(epoch+float(i)/train_steps_per_epoch)
            # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
            if i * TRAINING_RATIO % 100 == 0:
                generator_loss_average.append(np.average(generator_loss, axis=0))
                generator_loss = []
            if i * TRAINING_RATIO % 100 == 0:
                discriminator_loss_average.append(np.average(discriminator_loss, axis=0))
                discriminator_loss = []

            generated_images = generator.predict(MC_input)
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
                except:
                    print 'Plotting not successful:', epoch, i

                #loss_discriminator = np.asarray(discriminator_loss)
                discriminator_loss_aver = np.asarray(discriminator_loss_average)
                generator_loss_aver = np.asarray(generator_loss_average)

                plt.plot(discriminator_loss_aver[:, 0], color='red', markersize=12, label=r'Total')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_total.pdf')
                plt.close()
                plt.clf()

                plt.plot(discriminator_loss_aver[:, 1] + discriminator_loss_aver[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_wasserstein.pdf')
                plt.close()
                plt.clf()

                plt.plot(discriminator_loss_aver[:, 3], color='royalblue', markersize=12, label=r'GradientPenalty', linestyle='dashed')
                plt.legend()
                plt.savefig(path_save + 'loss_discriminitive_gradientpenality.pdf')
                plt.close()
                plt.clf()

                plt.plot(generator_loss_aver[:], label='generative loss')
                plt.legend()
                # plt.show()
                plt.savefig(path_save + 'loss_generative.pdf')
                plt.close()
                plt.clf()

                # np.savetxt(path_save + 'epoch_counter_fine.txt', (np.array(epoch_counter_fine)), delimiter=',')
                np.savetxt(path_save + 'discriminator_loss.txt', (np.array(discriminator_loss_average)), delimiter=',')
                # np.savetxt(path_save + 'epoch_counter.txt', (np.array(epoch_counter)), delimiter=',')
                np.savetxt(path_save + 'generator_loss.txt', (np.array(generator_loss_average)), delimiter=',')

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


        generator.save_weights(path_save + "GAN/generator_weights-" + str(epoch) + ".hdf5")
        discriminator.save_weights(path_save + "GAN/discriminator_weights-" + str(epoch) + ".hdf5")
        print 'Plotting'
        # - Plot the loss of discriminator and generator as function of iterations

generator.save(path_save + "GAN/generator-000" + ".hdf5")
discriminator.save(path_save + "GAN/discriminator-000" + ".hdf5")

generator.save_weights(path_save + "GAN/generator_weights-000" + ".hdf5")
discriminator.save_weights(path_save + "GAN/discriminator_weights-000" + ".hdf5")

train_for_n(epochs=50, batch_size=batch_size)