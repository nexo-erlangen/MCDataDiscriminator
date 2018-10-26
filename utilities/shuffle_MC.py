#!/usr/bin/env python

import numpy as np
import h5py
import argparse
import random

import os
from os import listdir
from os.path import isfile,join

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-out', dest='folderOUT', help='folderOUT Path')
parser.add_argument('-in' , dest='folderIN' , nargs=2, help='folderIN Paths')
#parser.add_argument('-file' , dest='filename' , help='folderIN Path')

args = parser.parse_args()
for i in range(len(args.folderIN)):
    args.folderIN[i]=os.path.join(args.folderIN[i],'')
args.folderOUT=os.path.join(args.folderOUT,'')

print
print 'Input Folder:\t', args.folderIN
print 'Output Folder:\t', args.folderOUT
print

files = {}
for folderIN in args.folderIN:
    files[folderIN] = [f for f in listdir(folderIN) if isfile(join(folderIN, f)) and '.hdf5' in f]
    print folderIN, '\t', len(files[folderIN]), 'files'

batchsize = 1
fletcher32 = True
shuffle = False
compression = ('gzip', 4)

counter = 0
minLength = min( [ len(f) for f in files.values() ] )

print
print 'number of files to merge (per input folder):\t' , minLength
print

for idx in range(minLength):
    eventInfo = {}
    fIN = {}
    for folder in files.keys():
        print "reading:", idx, "\t", folder, files[folder][idx]
        fIN[folder] = h5py.File(folder + str(files[folder][idx]), "r")
        eventInfo[folder] = {}
        for key in fIN[folder].keys():
            # if key in ['wfs']: continue
            eventInfo[folder][key] = np.asarray(fIN[folder][key])
        fIN[folder].close()

    eventInfoNew = {}
    for key in eventInfo.values()[0].keys():
        eventInfoNew[key] = np.concatenate([ eventInfo[folder][key] for folder in files.keys() ])

    NumEvents = list(set( [ len(eventInfoNew[key]) for key in eventInfoNew.keys() ] ))
    print 'shuffling:\tgenerating mask for %d events'%(NumEvents[0])
    if len(NumEvents) != 1 : print 'event numbers not equal'
    shuffleIdx = np.arange(NumEvents[0])
    random.shuffle(shuffleIdx)
    print 'shuffling:\t', shuffleIdx

    print 'shuffling:\tapplying mask'
    for key in eventInfoNew.keys():
        eventInfoNew[key] = eventInfoNew[key][shuffleIdx]

    fileOUT = args.folderOUT + str(counter) + "-shuffled.hdf5"
    print "creating:\t", fileOUT, "\t\t" , eventInfoNew['wfs'].shape
    chunks_wfs = (batchsize,) + eventInfoNew['wfs'].shape[1:] if compression[0] is not None else None
    # chunks_wfs = None
    print "creating:\tusing chunksize:\t", chunks_wfs

    fOUT = h5py.File(fileOUT, "w")
    fOUT.create_dataset("wfs", data=eventInfoNew['wfs'], dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in eventInfoNew.keys():
        if key in ['wfs']: continue
        fOUT.create_dataset(key, data=eventInfoNew[key], dtype=np.float32)
    fOUT.close()

    counter += 1
    print
    # if idx >=1: break

