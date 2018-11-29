#how to read files, and use them as input
import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_MC_P2/"

result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory





for i in range (1):
    fIN = h5py.File(path + str(i) + ".hdf5", "r") #reads just the file
    for j in range (20):
        plt.imshow(np.reshape(fIN['wfs'][j][0], (38, 350)), aspect='auto')
        plt.show()
    continue
    for j in range (38):
        plt.plot(fIN['wfs'][0][0][j]+j*20)
        #see note
    plt.show()
'''
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
