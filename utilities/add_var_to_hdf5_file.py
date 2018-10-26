#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility function to crop images to ROI."""

import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import h5py

batchsize = 1
fletcher32 = True
shuffle = False
compression = ('gzip', 4)

def main():
    # folderBDT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/bdt_combined_hdf5/'
    folderBDT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/bdt_combined_hdf5_v2/'
    folderDATA = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/'

    # folderIN = folderDATA + 'mixed_WFs_Uni_MC_P2/'
    # folderOUT = folderDATA + 'mixed_WFs_Uni_MC_P2-bdt/'
    # file_bdt = {}
    # file_bdt[0] = folderBDT + 'gamma_withBDt.hdf5'
    # file_bdt[1] = folderBDT + 'bb0nE_withBDt.hdf5'

    # folderIN = folderDATA + 'U238_WFs_AllVessel_MC_P2/'
    # folderOUT = folderDATA + 'U238_WFs_AllVessel_MC_P2-bdt/'
    # file_bdt = {}
    # file_bdt[0] = folderBDT + 'AllVessel_U238_withBDt.hdf5'

    # folderIN = folderDATA + 'Th232_WFs_AllVessel_MC_P2/'
    # folderOUT = folderDATA + 'Th232_WFs_AllVessel_MC_P2-bdt/'
    # file_bdt = {}
    # file_bdt[0] = folderBDT + 'AllVessel_Th232_withBDt.hdf5'

    folderIN = folderDATA + 'bb0n_WFs_Uni_MC_P2/'
    folderOUT = folderDATA + 'bb0n_WFs_Uni_MC_P2-bdt/'
    file_bdt = {}
    file_bdt[1] = folderBDT + 'bb0n_withBDt.hdf5'

    print
    print 'Input Folder:\t', folderIN
    print 'Output Folder:\t', folderOUT
    print

    files = [f for f in listdir(folderIN) if isfile(join(folderIN, f)) and '.hdf5' in f]

    print 'Number of Files:\t', len(files)
    print

    data_bdt = {}
    for id, file in file_bdt.items():
        print 'reading BDT file:\t', file
        data_bdt[id] = {}
        f = h5py.File(str(file), "r")
        for key in f.keys():
            data_bdt[id][key] = np.asarray(f[key])
        f.close()
        print 'Number events in file:\t', data_bdt[id].values()[0].shape[0]
        print

    for file in files:
        print 'adjusting file:\t\t', file
        add_var_to_file(folderIN+file, folderOUT+file, data_bdt=data_bdt)
        # break

def add_var_to_file(fileIN, fileOUT, data_bdt):
    fIN = h5py.File(fileIN, "r")
    fOUT = h5py.File(fileOUT, "w")

    fIN_RunNum = np.asarray(fIN.get('MCRunNumber'))
    fIN_EventNum = np.asarray(fIN.get('MCEventNumber'))
    fIN_ID = np.asarray(fIN.get('ID'))
    fIN_NumEvents = len(fIN_EventNum)


    keyID = {}
    keyID['disc_ss_dnn'] = 'BDT-DNN'

    # keyID = {}
    # keyID['numColl'] = 'CCNumCollWires'
    # keyID['risetime'] = 'CCRisetime'
    # keyID['maxV'] = 'CCMaxVFraction'
    # keyID['stand'] = 'CCStandoff'
    # keyID['event_sizeV'] = 'MCEventSizeV'
    # keyID['event_sizeU'] = 'MCEventSizeU'
    # keyID['event_sizeR'] = 'MCEventSizeR'
    # keyID['event_sizeZ'] = 'MCEventSizeZ'
    # keyID['disc_ss_std'] = 'BDT-SS-Std'
    # keyID['disc_ss_noStand'] = 'BDT-SS-Uni'
    # keyID['bdt_ss'] = 'BDT-SS'
    # keyID['bdt_ss_withV'] = 'BDT-SS-V'
    # keyID['bdt_ss_noStand'] = 'BDT-SS-NoStandoff'
    # keyID['bdt_all'] = 'BDT-SSMS'

    # keySkip = ['bdt_bins', 'energy', 'roc_disc_ss_std', 'roc_disc_ss_noStand', 'mult']

    # print fIN.keys()
    # print data_bdt[0].keys()
    #
    # print fIN.values()[0].size
    # for i in data_bdt.keys():
    #     print data_bdt[i]['dnn_var'].size, data_bdt[i]['runNum'].size

    fOUT_new = {}
    for key in keyID:
        if key not in data_bdt.values()[0].keys():
            raise ValueError('strange key: %s'%key)
        fOUT_new[keyID[key]] = np.zeros((fIN_NumEvents,), dtype=np.float32)

    for i in xrange(len(fIN_EventNum)):
        id = fIN_ID[i]
        index = np.where(np.logical_and(fIN_RunNum[i] == data_bdt[id]['runNum'],
                                        fIN_EventNum[i] == data_bdt[id]['eventNum']))[0][0]

        for key in keyID:
            fOUT_new[keyID[key]][i] = data_bdt[id][key][index]

    wfs = np.asarray(fIN.get('wfs'))
    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None

    fOUT.create_dataset('wfs', data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']: continue
        elif key in fOUT_new.keys(): continue
        temp = np.asarray(fIN.get(key))
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

    for key in fOUT_new.keys():
        fOUT.create_dataset(key, data=fOUT_new[key], dtype=np.float32)

    fIN.close()
    fOUT.close()

def add_var_to_predicted_file(fileIN, fileOUT, data_bdt):  # TODO set up properly!
    fIN = h5py.File(fileIN, "r")
    fOUT = h5py.File(fileOUT, "w")

    print fIN.keys()

    print data_bdt[0].keys()
    # exit()

    fIN_RunNum = np.asarray(fIN.get('MCRunNumber'))
    fIN_EventNum = np.asarray(fIN.get('MCEventNumber'))
    fIN_DNNPred = np.asarray(fIN.get('DNNPredClass'))
    fIN_ID = np.asarray(fIN.get('ID'))
    fIN_NumEvents = len(fIN_EventNum)

    keyID = {}
    keyID['disc_ss_dnn'] = 'BDT-DNN'

    fOUT_new = {}
    for key in keyID:
        if key not in data_bdt.values()[0].keys():
            raise ValueError('strange key: %s' % key)
        fOUT_new[keyID[key]] = np.zeros((fIN_NumEvents,), dtype=np.float32)

    for i in data_bdt.keys():
        print data_bdt[i]['dnn_var'].size, data_bdt[i]['runNum'].size
    # exit()

    for i in xrange(len(fIN_EventNum)):
        id = fIN_ID[i]
        index = np.where((fIN_RunNum[i] == data_bdt[id]['runNum']) & \
                         (fIN_EventNum[i] == data_bdt[id]['eventNum']))  # & \
        # (fIN_DNNPred[i] == data_bdt[id]['dnn_var'])) #[0][0]
        print id, index[0], fIN['DNNTrueClass'][index[0][0]], data_bdt[id]['dnn_var'][index[0][0]]
        raw_input('')
        continue

        for key in data_bdt[id].keys():
            if key in ['runNum', 'eventNum']:
                continue
            elif key in keySkip:
                continue
            fOUT_new[keyID[key]][i] = data_bdt[id][key][index]

    exit()

    wfs = np.asarray(fIN.get('wfs'))
    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None

    fOUT.create_dataset('wfs', data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs,
                        compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']:
            continue
        elif key in fOUT_new.keys():
            continue
        temp = np.asarray(fIN.get(key))
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

    for key in fOUT_new.keys():
        fOUT.create_dataset(key, data=fOUT_new[key], dtype=np.float32)

    fIN.close()
    fOUT.close()



if __name__ == '__main__':
    main()