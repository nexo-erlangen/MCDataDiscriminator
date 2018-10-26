#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile,join
from scipy.optimize import curve_fit
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator/')
import utilities.generator as gen

def exponential(x, par0, par1):
    return par0 * np.exp(x/par1)

def linear(x, par0, par1):
    return x*par0 + par1

def exponentiallin(x, par0, par1, par2):
    return par0 * np.exp(x/par1) + par2

def exponentials(x, par0, par1, par2):
    return par0 * ( np.exp(x/par1) + np.exp(x/par2) )

def exponentialpot2(x, par0, par1, par3):
    return par0 * (np.power(x-par3,2.0)) * np.exp((x-par3)/par1)

def exponentialpot3(x, par0, par1, par3):
    return par0 * (np.power(x-par3,3.0)) * np.exp((x-par3)/par1)

folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/gamma_WFs_Uni_MC/'
folderINold = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/gamma_WFs_Uni_MC-3/'
folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Plots/'

files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
filesOLD = [os.path.join(folderINold, f) for f in os.listdir(folderINold) if os.path.isfile(os.path.join(folderINold, f))]
print files
print filesOLD

EventInfo = gen.read_EventInfo_from_files(files)
EventInfoOLD = gen.read_EventInfo_from_files(filesOLD)
print 'current ', len(EventInfo.values()[0])
print 'old ', len(EventInfoOLD.values()[0])

data = np.asarray([EventInfo['QValue']])
dataOLD = np.asarray([EventInfoOLD['QValue']])

plt.ion()
hist, bin_edges = np.histogram(data/1.e3, bins=50, range=(1.0, 3.0), density=True)
histOLD, bin_edges = np.histogram(dataOLD/1.e3, bins=50, range=(1.0, 3.0), density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

coeff = [2.e0, -0.9]
for i in range(5):
    try:
        coeff, var_matrix = curve_fit(exponential, bin_centres, hist, p0=coeff)
        coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
    except:
        print 'fit did not work'

print 'EXP'
print coeff
print coeff_err
print abs(coeff_err/coeff)*100.
print '======================================='

coefflin = [-4.e-1, 1.1]
for i in range(5):
    try:
        coefflin, var_matrix = curve_fit(linear, bin_centres, hist, p0=coefflin)
        coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
    except:
        print 'fit did not work'

print 'LIN'
print coefflin
print coeff_err
print abs(coeff_err/coefflin)*100.
print '======================================='


plt.clf()
plt.step(bin_centres, hist, where='mid', label='current')
plt.step(bin_centres, histOLD, where='mid', label='old')

plt.plot(bin_centres, exponential(bin_centres, *coeff), label='Exp')
plt.plot(bin_centres, linear(bin_centres, *coefflin), label='Lin')


# plt.gca().set_yscale('log')
plt.xlim(xmin=0.95, xmax=3.05)
plt.legend(loc='best')
# plt.ylim(ymin=1.e-6, ymax=1.e-2)
plt.show()
plt.draw()
raw_input('')








