#how to read files, and use them as input
import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/mixed_WFs_S5_Th228_P2/"

result=[]
no_files = len(os.listdir( path )) #this function gives the number of files in the directory

        
def get_wfs(fIN, event, index):
    #print fIN['wfs'].shape
    y_temp = fIN['wfs'][event]
    if index == 'positive':
        return y_temp[2]
    elif index == 'negative':
        return y_temp[0]
    else:
        print 'get_wfs got a problem'
                                                                                    



for i in range (1):
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r") #reads just the file
    column = []
    #print fIN.keys()
    #print fIN.values()[0].shape
    for event in xrange(fIN.values()[0].shape[0]):
        #print fIN.values()[0].shape[0] #result is 8000 the number of lines
        #print fIN['CCPosZ'][event].shape #array that shows the line and what's inside every line (8000,38, 350, 1) or so...
        z_temp = fIN['CCPosZ'][event]
        wfs = fIN['wfs'][event][[0, 2]] #get_wfs(fIN, event) #, index='positive'))
        if all (z>=0. for z in z_temp):
            wfs = wfs[1]
            #column.append(get_wfs(fIN, event, index='positive'))
            #print column.shape
            #plt.imshow(np.reshape(column[column.shape[0]], (38, 350)), aspect='auto')
            #plt.show()
            print 'showing TPC: NEGATIVE', fIN['IsMC'][event]
            plt.imshow(wfs[...,0], aspect='auto')
            plt.show()
        elif all (z<=0. for z in z_temp):
            wfs = wfs[0]
            print 'showing TPC: POSITIVE', fIN['IsMC'][event]
            plt.imshow(wfs[...,0], aspect='auto')
            plt.show()
            #column.append(get_wfs(fIN, event, index='negative'))
            #print column.shape
            #plt.imshow(np.reshape(column[column.shape[0]], (38, 350)), aspect='auto')
            #plt.show()
        else:
            pass
        if event == 10:
            exit()
                
#add to program
'''
path = "/home/vault/capm/mppi060h/MCDataDiscriminator/Data/mixed_WFs_S5_Th228_P2/"

no_files = len(os.listdir( path )) #this function gives the number of files in the directory


for i in range (len(os.listdir( path ))):  #this function gives the number of files in the directory
    fIN = h5py.File(path + str(i) + "-shuffled.hdf5", "r") #reads just the file
    for event in xrange(fIN.values()[0].shape[0]):
        z_temp = fIN['CCPosZ'][event]
        wfs = fIN['wfs'][event][[0, 2]] #get_wfs(fIN, event) #, index='positive'))
        if all (z>=0. for z in z_temp):
            wfs = wfs[1]
       elif all (z<=0. for z in z_temp):
            wfs = wfs[0]
        else:
            pass
 
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
