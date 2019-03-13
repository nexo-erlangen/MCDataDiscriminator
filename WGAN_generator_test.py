import argparse
import os, sys, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utilities.generator import *
from matplotlib import colors

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
path_generator = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_01_22-12_32_04/GAN/'
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
# gen_real = generate_batches_from_files(files, batch_size , wires='V', class_type='binary_bb_gamma', f_size=None,
#                                        select_dict=real, yield_mc_info=0)

def load_generator():
    import keras as ks
    generator = ks.models.load_model(path_generator + 'generator-000.hdf5')
    generator.load_weights(path_generator + 'generator_weights-27.hdf5')
    generator.summary()
    return generator

generator = load_generator()

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