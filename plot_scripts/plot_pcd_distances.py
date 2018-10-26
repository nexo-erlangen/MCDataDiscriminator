#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from os import listdir
from os.path import isfile,join
import h5py

# folder = '/home/hpc/capm/sn0515/UVWireRecon/'
# file = folder+'test.dat'
# data = np.loadtxt(file)
# print data.shape

folder = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/UniformGamma_ExpWFs_MC_SS/'
files = [f for f in listdir(folder) if isfile(join(folder, f)) and '.hdf5' in f]

folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Plots/'

for index, filename in enumerate(files):
    # if index>=5: break
    fIN = h5py.File(folder + str(filename), "r")
    if index == 0:
        data = np.asarray([fIN['MCEnergy'][:,0],fIN['G4NumActivePCD'],fIN['G4MaxDistActivePCD'],fIN['G4MaxDistAllPCD']])
    else:
        data = np.concatenate((data, np.asarray([fIN['MCEnergy'][:,0],fIN['G4NumActivePCD'],fIN['G4MaxDistActivePCD'],fIN['G4MaxDistAllPCD']])), axis=1)
    # print data.shape
    fIN.close()
print data.shape

# mask0 = data[0,:]>=550.
# data = data[:,mask0]
# mask00 = data[0,:]<=3500.
# data = data[:,mask00]
# print data.shape
# mask1 = data[2,:]<=5.
# data = data[:,mask1]
# mask2 = data[3,:]<=6.
# data = data[:,mask2]
# print data.shape


plt.ion()

activePCD = 15
maxDist = 6 #2
pixelsize = .1
minE = 500
maxE = 3550

plt.clf()
plt.hist2d(data[0], data[2], bins=[np.arange(minE,maxE,(maxE-minE)/100),np.arange(0,maxDist,pixelsize)], cmin=1)
plt.ylabel('Max distance between active PCDs [mm]')
plt.xlabel('Energy [keV]')
plt.savefig(folderOUT+'heatmap_dist_active_PCDs.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist2d(data[0], data[3], bins=[np.arange(minE,maxE,(maxE-minE)/100),np.arange(0,maxDist,pixelsize)], cmin=1)
plt.ylabel('Max distance between all PCDs [mm]')
plt.xlabel('Energy [keV]')
plt.savefig(folderOUT+'heatmap_dist_all_PCDs.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist2d(data[2], data[3], bins=[np.arange(0,maxDist,pixelsize),np.arange(0,maxDist,pixelsize)], cmin=1)
plt.xlabel('Max distance between active PCDs [mm]')
plt.ylabel('Max distance between all PCDs [mm]')
plt.savefig(folderOUT+'heatmap_dist_all_active_PCDs.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist(data[0], bins=100)
plt.xlabel('Energy [keV]')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist2d(data[1], data[2], bins=[range(activePCD),np.arange(0,maxDist,pixelsize)], cmin=1)
plt.xlabel('Active PCDs')
plt.ylabel('Max distance between active PCDs [mm]')
plt.savefig(folderOUT+'heatmap.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist(data[1],bins=range(activePCD))
plt.xlabel('Active PCDs')
plt.savefig(folderOUT+'activePCD.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

plt.clf()
plt.hist(data[2],bins=np.arange(0,maxDist,pixelsize), cumulative=True)
plt.xlabel('Max distance between active PCDs [mm]')
plt.savefig(folderOUT+'MaxDistActPCD-Cum.png', bbox_inches='tight')
plt.show()
plt.draw()
raw_input('')

# plt.clf()
# plt.hist(data[2],bins=np.arange(0,maxDist,pixelsize), cumulative=False)
# plt.xlabel('Max distance between active PCDs [mm]')
# plt.savefig(folder+'MaxDistActPCD.png', bbox_inches='tight')
# plt.show()
# plt.draw()
# raw_input('')



# ----------------------------------------------------------
# Math Functions
# ----------------------------------------------------------
def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def erf(x, mu, sigma, B):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)