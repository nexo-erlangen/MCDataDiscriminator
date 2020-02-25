import os, sys, time
import numpy as np
import h5py
import keras as ks
import matplotlib.pyplot as plt
# from utilities.generator import *
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
# path_load = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_06_07-14_54_55/GAN/'
# path_load = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_06_11-11_21_57/GAN/'
path_load = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_10_02-12_08_22/GAN/'


last_epoch = 4

batch_size=40
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = batch_size
NCR = 5


######################
                    ##
# status = 'test'   ##
status = 'train'  ##
                    ##
######################



path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/plots_WGAN_UV"

try:
    os.mkdir(path_save)
    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/plots_WGAN_UV/"
except:
    path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/plots_WGAN_UV/"

#
#
#
# if status == 'test':
#
#     path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + 'Dummy/'
#     path_model = path_save
#
# elif status == 'train':
#
#     now = time.strftime("%Y_%m_%d-%H_%M_%S")
#     path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/" + str(now)
#     os.mkdir(path_save)
#     path_save = path_save + "/"
#     path_model = path_save + "GAN"
#     os.mkdir(path_model)
#     path_model = path_model + "/"
# else:
#     raise ValueError('Error with status: %s' % status)


result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory
files = []
for i in range(15):
    files.append(path + str(i) + "-shuffled.hdf5")
fake = {'IsMC': [0]}
real = {'IsMC': [1]}

# gen_real = generate_batches_from_files(files, batch_size , wires='UV', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=real, yield_mc_info=0)
#
# files_test = []
# for i in range(15, no_files):
#     files_test.append(path + str(i) + "-shuffled.hdf5")
# gen_fake_test = generate_batches_from_files(files_test, batch_size, wires='UV', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=fake, yield_mc_info=0)
# gen_real_test = generate_batches_from_files(files_test, batch_size , wires='UV', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=real, yield_mc_info=0)


def encode_targets(y_dict, batchsize, class_type=None):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. binary_bb_gamma
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    if class_type == None:
        train_y = np.zeros(batchsize, dtype='float32')
    elif class_type == 'binary_bb_gamma':
        from keras.utils import to_categorical
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['IsMC']  # event ID (0: Data, 1: MC)
        train_y = to_categorical(train_y, 2)  # convert to one-hot vectors
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')
    return train_y

def getNumEvents(files):
    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y,files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['EventNumber'].shape[0]
        f.close()
    return counter

def select_events(data_dict, select_dict={}, shuffle=False):
    """
    Encodes the labels (classes) of the images.
    :param dict data_dict: Dictionary that contains ALL event class information for the events.
    :param dict select_dict: Dictionary that contains keys to select events with their values (or low/up limit for range selections).
    :param bool shuffle: Boolean to specify whether the index output list should be shuffled.
    :return: list lst: List that holds the events indices that pass the given selection criteria.
    """

    mask = np.ones(data_dict.values()[0].shape[0], dtype=bool)
    for key, value in select_dict.items():
        if key not in data_dict.keys(): raise ValueError('Key not in data dict: %s'%(key))
        if isinstance(value, list) and len(value) == 1:
            mask = mask & (data_dict[key] == value[0])
        elif isinstance(value, list) and len(value) == 2:
            mask = mask & (data_dict[key] >= value[0]) & (data_dict[key] < value[1])
        else:
            raise ValueError('Key/Value pair is strange. key: %s . value: %s)'%(key, value))
    lst = np.squeeze(np.argwhere(mask))
    if shuffle: random.shuffle(lst)
    return lst

#------------- Function used for supplying images to the GPU -------------#
def generate_batches_from_files(files, batchsize, wires=None, class_type=None, f_size=None, select_dict={}, yield_mc_info=0):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param int batchsize: Size of the batches that should be generated.
    :param str class_type: String identifier to specify the exact target variables. i.e. 'binary_bb_gamma'
    :param int/None f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param int yield_mc_info: Specifies if mc-infos should be yielded. 0: Only Waveforms, 1: Waveforms+MC Info, 2: Only MC Info
                               The mc-infos are used for evaluation after training and testing is finished.
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info depending on yield_mc_info).
    """

    import keras as ks

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    if wires == 'U':    wireindex = [0, 2]
    elif wires == 'V':  wireindex = [1, 3]
    elif wires in ['UV', 'UV_S']: wireindex= [0, 1, 2, 3]#slice(4) #pass #wireindex = [0, 1, 2, 3]
    else: raise ValueError('passed wire specifier need to be U/V/UV')

    eventInfo = {}
    while 1:
        # random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)
                # warnings.warn( 'f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                #     'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

            # filter the labels we don't want for now
            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])
            ys = encode_targets(eventInfo, f_size, class_type)

            lst = select_events(eventInfo, select_dict=select_dict, shuffle=False)
            # lst = np.arange(0, f_size, batchsize)
            # random.shuffle(lst)
            if wires == 'V' or wires == 'U':
                if not yield_mc_info in [-1,2]:
                    xs = np.asarray(f['wfs'])[:, wireindex]
                    z_temp = f['CCPosZ'][:]
                    for event in xrange(z_temp.shape[0]):
                        if all(z >= 0. for z in z_temp[event]):
                            pass
                            # xs[event, 1] = xs[event, 0]
                        elif all(z <= 0. for z in z_temp[event]):
                            xs[event, 0] = xs[event, 1]
                        else:
                            xs[event, 0] = xs[event, 1]
                        # print xs.shape
                    xt = xs[:,0]

            elif wires == 'UV':
                if not yield_mc_info in [-1, 2]:
                    xs = np.asarray(f['wfs'])[:, wireindex]
                    z_temp = f['CCPosZ'][:]
                    for event in xrange(z_temp.shape[0]):
                        # if event%100==0: print event, z_temp.shape[0]
                        if all(z >= 0. for z in z_temp[event]):
                            pass
                            # xt = xs[event, [0, 1]]
                            # xs[event, 2] = xs[event, 0]
                            # xs[event, 3] = xs[event, 1]
                        elif all(z <= 0. for z in z_temp[event]):
                            # xt = xs[event, [2, 3]]
                            xs[event, 0] = xs[event, 2]
                            xs[event, 1] = xs[event, 3]
                        else:
                            # xt = xs[event, [2, 3]]
                            xs[event, 0] = xs[event, 2]
                            xs[event, 1] = xs[event, 3]
                        # print xs.shape
                    xt = xs[:, [0, 1]]
                # xs = np.reshape(xs, (xs.shape[0], 76, 350, -1))

            elif wires == 'UV_S':
                if not yield_mc_info in [-1, 2]:
                    xs = np.asarray(f['wfs'])[:, wireindex]
                    z_temp = f['CCPosZ'][:]
                    for event in xrange(z_temp.shape[0]):
                        # if event%100==0: print event, z_temp.shape[0]
                        if all(z >= 0. for z in z_temp[event]):
                            pass
                            # xt = xs[event, [0, 1]]
                            # xs[event, 2] = xs[event, 0]
                            # xs[event, 3] = xs[event, 1]
                        elif all(z <= 0. for z in z_temp[event]):
                            # xt = xs[event, [2, 3]]
                            xs[event, 0] = xs[event, 2]
                            xs[event, 1] = xs[event, 3]
                        else:
                            # xt = xs[event, [2, 3]]
                            xs[event, 0] = xs[event, 2]
                            xs[event, 1] = xs[event, 3]
                        # print xs.shape
                    xt = xs[:, [0, 1]]
                xt = np.reshape(xt, (xt.shape[0], 38, 350, 2))

            for i in np.arange(0, lst.size, batchsize):
                batch = sorted(lst[i: i + batchsize])

                # if len(batch) != batchsize: continue

                if not yield_mc_info in [-1,2]:
                    xs_i = xt[batch]
                    ys_i = ys[batch]

            # for i in lst:
            #     if not yield_mc_info == 2:
            #         # if wires in ['U', 'V', 'UV', 'U+V']:
            #         #     z_temp = f['CCPosZ'][i: i + batchsize]
            #         xs_i = f['wfs'][i: i + batchsize, wireindex]  # get_wfs(fIN, event) #, index='positive'))
            #         #     for event in xrange(z_temp.shape[0]):
            #         #         if all(z >= 0. for z in z_temp[event]):
            #         #             xs_i[event, 0] = np.zeros(xs_i[event, 0].shape)
            #         #         elif all(z <= 0. for z in z_temp[event]):
            #         #             xs_i[event, 1] = np.zeros(xs_i[event, 1].shape)
            #         #         else:
            #         #             xs_i[event] = np.zeros(xs_i[event].shape)
            #         # else: raise ValueError('passed wire specifier need to be U/V/UV')
            #         # xs_i = np.swapaxes(xs_i, 0, 1)
            #         # xs_i = np.swapaxes(xs_i, 2, 3)
            #         xs_i = np.reshape(xs_i, (batchsize, 76, 350, -1))
            #         ys_i = ys[ i : i + batchsize ]

                if   yield_mc_info == 0:    yield (xs_i, ys_i)
                elif yield_mc_info == 1:    yield (xs_i, ys_i) + ({ key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() }
                else:   raise ValueError("Wrong argument for yield_mc_info (0/1/2)")
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize


gen_fake = generate_batches_from_files(files, batch_size, wires='UV', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)



MC_input, y_MC = gen_fake.next()
# X_train, y_train = gen_real.next()


def load_generator():
    generator = ks.models.load_model(path_load + 'generator-000.hdf5')
    generator.load_weights(path_load + 'generator_weights-' + str(last_epoch) + '.hdf5')
    print '\nGenerator'
    generator.summary()
    return generator

def load_discriminator():
    discriminator = ks.models.load_model(path_load + 'discriminator-000.hdf5')
    discriminator.load_weights(path_load + 'discriminator_weights-' + str(last_epoch) + '.hdf5')
    print '\nCritic'
    discriminator.summary()
    return discriminator

def plot_structure(model):
    try: # plot model, install missing packages with conda install if it throws a module error
        ks.utils.plot_model(model, to_file=path_save + 'plot_%s.png'%(str(model)),
                            show_shapes=True, show_layer_names=False)
    except OSError:
        print '\nCould not produce plot_%s.png'%(str(model))



generator = load_generator()
critic = load_discriminator()

# models = [generator, critic]
# for model in models:
#     plot_structure(model)

print '\nModels plotted'

generated_images = generator.predict([MC_input[:,0,:,:,:], MC_input[:,1,:,:,:]])
# generated_images = MC_input

print '\nPredict complete'

generated_images[0] = np.reshape(generated_images[0], (batch_size, 38, 350))
generated_images[1] = np.reshape(generated_images[1], (batch_size, 38, 350))
MC_input = np.reshape(MC_input, (batch_size, 2, 38, 350))
color = 'black'

for i in range(batch_size):
    for j in range(MC_input.shape[2]):

        plt.subplot(322)
        plt.title('U Wire')
        plt.plot(MC_input[i, 0][j] + j * 20, color=color, linewidth=0.5)
        plt.yticks([])
        ax2 = plt.twinx()
        ax2.set_ylabel('Original MC')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 350)
        plt.subplot(324)
        plt.plot(generated_images[0][i, j] + j * 20, color=color, linewidth=0.5)
        plt.yticks([])
        ax2 = plt.twinx()
        ax2.set_ylabel('Modified MC')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 350)
        plt.subplot(326)
        plt.plot(np.subtract(generated_images[0][i, j], MC_input[i, 0][j]) + j * 20, color=color,
                 linewidth=0.5)
        plt.yticks([])
        plt.xlabel('Time [$\mu$s]')
        ax2 = plt.twinx()
        ax2.set_ylabel('Mod - Orig')
        plt.xlabel('Time [$\mu$s]')
        plt.yticks([])
        plt.xlim(0, 350)

        plt.subplot(321)
        plt.plot(MC_input[i, 1][j] + j * 20, color=color, linewidth=0.5)
        plt.title('V Wire')
        plt.ylabel('Channel')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 350)
        plt.subplot(323)
        plt.plot(generated_images[1][i, j] + j * 20, color=color, linewidth=0.5)
        plt.ylabel('Channel')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 350)
        plt.subplot(325)
        plt.plot(np.subtract(generated_images[1][i, j], MC_input[i, 1][j]) + j * 20, color=color,
                 linewidth=0.5)
        plt.ylabel('Channel')
        plt.xlabel('Time [$\mu$s]')
        plt.yticks([])
        plt.xlim(0, 350)
    plt.savefig(path_save + 'wfs_' + '_' + str(i) + '_plot_3D.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()



