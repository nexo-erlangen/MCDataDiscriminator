#how to read files, and use them as input
import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

path = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_03_07-14_50_27/discriminator_loss.txt'
# path = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/2019_03_07-14_48_49/discriminator_loss.txt'

loss = np.loadtxt(path, delimiter=',')
loss_average = []

def average(i):
    a0 = np.average(loss[i*1000:(i+1)*1000, 0])
    a1 = np.average(loss[i*1000:(i+1)*1000, 1])
    a2 = np.average(loss[i*1000:(i+1)*1000, 2])
    a3 = np.average(loss[i*1000:(i+1)*1000, 3])
    return a0, a1, a2, a3

for i in range(loss.shape[0]//1000):
    loss_average.append(average(i))
loss_average = np.array(loss_average)

print loss_average.shape

# plt.plot(loss_average[:,0], marker='.', linestyle='None', markersize=1)
plt.plot(loss_average[:,1] + loss_average[:,2], marker='o', linestyle='None', markersize=2)
# plt.axhline(0, color='black', linewidth=1)
plt.ylim(-100, 0)
# plt.xlim(14000, 16000)
plt.show()

print loss.shape
exit()


a = 10
y = np.arange(float(a))

x = y/10.

plt.plot(x, y)
plt.show()


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
