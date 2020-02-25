#how to read files, and use them as input
import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
from utilities.generator import *

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"
path_save = "/home/vault/capm/mppi060h/MCDataDiscriminator/pics_thesis/"
batch_size = 40


files = []
for i in range(1):
    files.append(path + str(i) + "-shuffled.hdf5")
fake = {'IsMC': [1]}
real = {'IsMC': [0]}
gen_fake = generate_batches_from_files(files, batch_size, wires='UV', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=fake, yield_mc_info=0)
gen_real = generate_batches_from_files(files, batch_size , wires='UV', class_type='binary_bb_gamma', f_size=None,
                                       select_dict=real, yield_mc_info=0)

MC_input, y_MC = gen_fake.next()
X_train, y_train = gen_real.next()

print MC_input.shape
print X_train.shape

MC_input = np.reshape(MC_input, (batch_size, 2, 38, 350))
X_train = np.reshape(X_train, (batch_size, 2, 38, 350))

for i in range(batch_size):
    for j in range(X_train.shape[2]):
        plt.subplot(211)
        plt.plot(X_train[i][1][j] + j * 20, color='black', linewidth=0.5)
        # plt.title('Data Signal')
        plt.ylabel('Amplitude w/ offset')
        # plt.xlabel('Time steps')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 350)
        ax2 = plt.twinx()
        ax2.set_ylabel('V wires')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(212)
        plt.plot(X_train[i][0][j] + j * 20, color='black', linewidth=0.5)
        # plt.title('V Wires')
        plt.ylabel('Amplitude w/ offset')
        plt.xlabel('Time [$\mu$s]')
        plt.yticks([])
        plt.xlim(0, 350)
        ax2 = plt.twinx()
        ax2.set_ylabel('U wires')
        plt.yticks([])

    plt.savefig(path_save + 'wfs_' + str(i) + 'plot.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()
    plt.clf()


# path_load = "/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_02_11-16_17_02_h48/"
#
# discriminator_acc = np.loadtxt(path_load + 'discriminator_acc.txt')
#
# discriminator_acc_mean = []
# for i in range(len(discriminator_acc)//1000):
#     discriminator_acc_mean.append(np.mean(discriminator_acc[i*100:(i+1)*100]))
#
# plt.plot(discriminator_acc_mean, color='green', label='discriminator accuracy', linewidth=0.5)#, marker='.', linestyle='None', markersize=1)
# plt.title('Discriminator Accuracy')
# plt.xlabel('Training Steps')
# plt.ylabel('Accuracy')
# plt.ylim(0.55, 0.65)
# #plt.legend()
# # plt.savefig(path_save + 'discriminator_acc_no-line.pdf')
# # plt.close()
# # plt.clf()
# plt.show()
#
# print len(discriminator_acc_mean)
#
#
#
# discriminator_acc_mean = []
# if len(discriminator_acc) >= traing_steps:
#     discriminator_acc_mean.append(np.mean(discriminator_acc))
#     discriminator_acc = []
#


'''
#write in files
#import sys

orig_stdout = sys.stdout
f = open('GAN_structure.txt', 'w')
sys.stdout = f

for i in range(2):
    print 'i = ', i

sys.stdout = orig_stdout
f.close()

'''







'''
path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"

no_files = len(os.listdir( path )) #this function gives the number of files in the directory
MC_input = []
DATA_input = []
eventInfo = dict()
batch_size = 32
wireindex = [1, 3]

for i in range (1):
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
    for key in fIN.keys():
        if key in ['wfs']: continue
        eventInfo[key] = np.asarray(fIN[key])
    ys = eventInfo['IsMC']
    noise_mask = ys == 1.
    real_mask = np.invert(noise_mask)
    #noise_mask = np.asarray(noise_mask)
    #real_mask = np.asarray(real_mask)
    for key in fIN.keys():
        if key not in ['wfs']: continue
        wfs = fIN['wfs'][:, wireindex]
        MC_input = wfs[noise_mask]
        DATA_input = wfs[rel_mask]
    lst = np.arange(0, MC_input.shape[0], batchsize)
    random.shuffle(lst)
    for i in lst:
        MC_input = fIN['wfs'][i: i + batch_size]
        DATA_input = fIN['wfs'][i: i + batch_size]
        MC_input = np.reshape(MC_input, (batch_size, 76, 350, -1))
        DATA_input = np.reshape(DATA_input, (batch_size, 76, 350, -1))
    print MC_input.shape, DATA_input.shape

'''

'''

            ys = encode_targets(eventInfo, f_size, class_type)
            ys = ks.utils.to_categorical(ys, 2) 

        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = eventInfo['IsMC'] 
        ys = train_y

    





                                                                                   
for i in range (1):
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
    for event in xrange(fIN.values()[0].shape[0]):
        if fIN['IsMC'][event] == 1.0:
            MC_input.append(fIN['wfs'][event])
        elif fIN['IsMC'][event] == 0.0:
            DATA_input.append(fIN['wfs'][event])
        else:
            print 'There is a problem with the reading data'
    print 'Data from file ', str(i+1), '/', str(no_files), ' acquired.'
MC_input = np.array(MC_input)
DATA_input = np.array(DATA_input)
wireindex = [1, 3]

MC_input = MC_input[:,wireindex,:,:,:]
DATA_input = DATA_input[:,wireindex,:,:,:]
MC_temp = []
DATA_temp = []
print 'Concatenating'
for i in range(MC_input.shape[0]):
    MC_temp.append(np.concatenate((MC_input[i][0],MC_input[i][1]),axis=0))
    DATA_temp.append(np.concatenate((DATA_input[i][0],DATA_input[i][1]),axis=0))    
MC_input = np.array(MC_temp)
DATA_input = np.array(DATA_temp)
print MC_input.shape, DATA_input.shape
exit()

'''


'''
ntrain = 1000
latent_dim = 100
noise_gen = np.random.uniform(0,1,size=[ntrain, latent_dim])

print noise_gen.shape

plt.imshow(noise_gen, aspect='auto')
plt.show()
'''
'''
path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/"

result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory
MC_input = []
DATA_input = []
                                                                                    
for i in range (1):
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r")
    for event in xrange(fIN.values()[0].shape[0]):
        if fIN['IsMC'][event] == 1.0:
            MC_input.append(fIN['wfs'][event])
        elif fIN['IsMC'][event] == 0.0:
            DATA_input.append(fIN['wfs'][event])
        else:
            print 'There is a problem with the reading data'
    print 'Data from file ', str(i+1), '/', str(no_files), ' acquired.'
MC_input = np.array(MC_input)
DATA_input = np.array(DATA_input)
wireindex = [1, 3]

MC_input = MC_input[:,wireindex,:,:,:]
DATA_input = DATA_input[:,wireindex,:,:,:]
MC_temp = []
DATA_temp = []
print 'Concatenating'
for i in range(MC_input.shape[0]):
    MC_temp.append(np.concatenate((MC_input[i][0],MC_input[i][1]),axis=0))
    DATA_temp.append(np.concatenate((DATA_input[i][0],DATA_input[i][1]),axis=0))    
MC_input = np.array(MC_temp)
DATA_input = np.array(DATA_temp)
print MC_input.shape, DATA_input.shape
exit()
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

print MC_input.shape  
print DATA_input.shape
exit()
frac_test = int(DATA_input.shape[0]*0.1)
X_test = DATA_input[:frac_test]
X_train = DATA_input[frac_test:]

'''
