import numpy as np
import keras as ks
import os
import h5py
import innvestigate
import innvestigate.utils as iutils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
#from keras import backend as K

folderRun = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/181106-1720/'
model_file = folderRun + 'models/model-000.hdf5'
weights_file = folderRun + 'models/weights-080.hdf5'
folderInput = '/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/'


folderOUT = folderRun + 'output_training/'
folderIN = '/home/vault/capm/mppi060h/MCDataDiscriminator/Data/Th228_WFs_S5_mixed_P2/'
folderRUNS = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/181106-1720/'
folderMODEL = 'models/'
num_weights = 0
sources = ['Th228']
wires ='V'
mode = 'valid'
var_targets = 'binary_bb_gamma'
batchsize = 1
events = 2000

endings = {}





def main():
    files = getFiles()
    executeCNN(files)




def getFiles():
    files = [ os.path.join(folderIN,f) for f in os.listdir(folderIN) ]
    return files

def plot_structure(model):
    try: # plot model, install missing packages with conda install if it throws a module error
        ks.utils.plot_model(model, to_file=folderRun + 'plot_model.png',
                            show_shapes=True, show_layer_names=False)
        print ('\n+++++++++ plot_model.png was created +++++++++')
    except OSError:
        print ('\nCould not produce plot_model.png')

def executeCNN(files):

    model = ks.models.load_model(model_file)
    model.load_weights(weights_file)
    model.summary()
    plot_structure(model)

    print ('Validate events')
    print (model)

    global folderRun
    # folderRun += "0validation/test/"
    folderRun += "0validation/test_dt/"
    # os.system("mkdir -p -m 770 %s " % (folderRun))
    print(folderRun)

    gen = generate_batches_from_files(files, batchsize=batchsize, wires=wires, class_type=var_targets, yield_mc_info=0)
    for i in range(10):
        wfs, y = next(gen)
        for mode in ['deep_taylor']:# ['smoothgrad', 'gradient', 'deconvnet', 'input_t_gradient', 'integrated_gradients', 'deep_taylor']:
            for layer in [-1, 0, 1]:
                analyzer(model, mode, wfs, y, i, layer)





def analyzer(model, mode, wfs, label, event, index):
    model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

    if index >= 0:
        gradient_analyzer = innvestigate.create_analyzer(mode, model_wo_sm, neuron_selection_mode="index")#, postprocess='abs')
        analysis = gradient_analyzer.analyze(wfs, index)
    else:
        gradient_analyzer = innvestigate.create_analyzer(mode, model_wo_sm)  # , postprocess='abs')
        analysis = gradient_analyzer.analyze(wfs)

    print(label)
    if label[0][0] == 0.0:
        label = 'class: MC event'
        label_file = 'MC'
    else:
        label = 'class: Real event'
        label_file = 'Data'
    range_x = [0, 350]
    range_y = [0, 38]
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    aspect = "auto"
    analysis /= np.max(analysis)
    wfs /= np.max(wfs)

    plt.clf()
    f, axarr = plt.subplots(2, 2, sharex='all', sharey='all')
    f.set_size_inches(w=15., h=7.)
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])#, sharey=ax1)
    ax3 = plt.subplot(gs[2])#, sharex=ax1)
    ax4 = plt.subplot(gs[3])#, sharex=ax2, sharey=ax3)
    #f.suptitle(label+', analyzer: Deep Taylor', fontsize=14)# layer: %i'%(index+1), fontsize=14)
    h1 = ax1.imshow(analysis[0].squeeze().T, extent=extent, interpolation='nearest',vmin=-1,vmax=1, cmap='seismic',
                    origin='lower', aspect=aspect)
    h2 = ax2.imshow(analysis[1].squeeze().T, extent=extent, interpolation='nearest',vmin=-1,vmax=1, cmap='seismic',
                    origin='lower', aspect=aspect)
    h3 = ax3.imshow(wfs[0].squeeze().T, extent=extent, interpolation='nearest', vmin=-1,vmax=1, cmap='seismic',
                    origin='lower', aspect=aspect)
    h4 = ax4.imshow(wfs[1].squeeze().T, extent=extent, interpolation='nearest', vmin=-1,vmax=1, cmap='seismic',
                    origin='lower', aspect=aspect)
    f.colorbar(h1, ax=ax1, shrink=0.6, ticks=[-1, 0, 1]).set_label(label=r'Deep Taylor', size=14)#, label=r'Deep Taylor', size=14 extend='both', )
    # f.colorbar(h1).set_label(label=r'Deep Taylor', size=14)
    f.colorbar(h2, ax=ax2, shrink=0.6, ticks=[-1, 0, 1]).set_label(label=r'Deep Taylor', size=14)#, label=r'Deep Taylor', size=14)
    # f.colorbar(h2).set_label(label=r'Deep Taylor', size=14)
    f.colorbar(h3, ax=ax3, shrink=0.6, ticks=[-1, 0, 1]).set_label(label=r'Input', size=14)#, label=r'Deep Taylor', size=14)
    # f.colorbar(h3).set_label(label=r'Deep Taylor', size=14)
    f.colorbar(h4, ax=ax4, shrink=0.6, ticks=[-1, 0, 1]).set_label(label=r'Input', size=14)#, label=r'Deep Taylor', size=14)
    # f.colorbar(h4).set_label(label=r'Deep Taylor', size=14)


    # ax1.get_xaxis().set_visible(False)
    ax1.set_title('TPC 1', fontsize=14)
    ax2.set_title('TPC 2', fontsize=14)
    ax1.set_xlim(range_x)
    ax1.set_ylim(range_y)
    ax2.set_xlim(range_x)
    ax2.set_ylim(range_y)
    ax3.set_xlim(range_x)
    ax3.set_ylim(range_y)
    ax4.set_xlim(range_x)
    ax4.set_ylim(range_y)

    ax1.set_ylabel(r'Channel', fontsize=14)#%s'%(mode), fontsize=14)
    ax2.set_ylabel(r'Channel', fontsize=14)
    ax3.set_ylabel(r'Channel', fontsize=14)  # %s'%(mode), fontsize=14)
    ax4.set_ylabel(r'Channel', fontsize=14)
    ax3.set_xlabel(r'Time [$\mu$s]', fontsize=14)
    ax4.set_xlabel(r'Time [$\mu$s]', fontsize=14)

    # plt.setp(ax2.get_yticklabels(), visible=False)
    # plot.imshow(analysis[0].squeeze(), origin='lower', cmap='seismic', interpolation='nearest')
    # plt.show()
    #print(folderRun+'%s_%s_%s.pdf'%(mode,  label_file, str(index+1)))
    plt.savefig(folderRun+'%s_%s_%s_%s.pdf'%(str(event), mode,  label_file, str(index+1)), bbox_inches='tight')

    return









def generate_batches_from_files(files, batchsize, wires=None, class_type=None, f_size=None, yield_mc_info=0):
    import keras as ks
    import random

    if isinstance(files, list): pass
    elif isinstance(files, str): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    if wires == 'U':    wireindex = [0, 2]
    elif wires == 'V':  wireindex = [1, 3]
    elif wires in ['UV', 'U+V']: wireindex= slice(4) #pass #wireindex = [0, 1, 2, 3]
    else: raise ValueError('passed wire specifier need to be U/V/UV')

    eventInfo = {}
    while 1:
        # random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)

            lst = np.arange(0, f_size, batchsize)
            # random.shuffle(lst)

            # filter the labels we don't want for now
            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])
            ys = encode_targets(eventInfo, f_size, class_type)
            ys = ks.utils.to_categorical(ys, 2) #convert to one-hot vectors

            for i in lst:
                if not yield_mc_info == 2:
                    if wires in ['U', 'V', 'UV', 'U+V']:
                        xs_i = f['wfs'][i: i + batchsize, wireindex]
                    else: raise ValueError('passed wire specifier need to be U/V/UV')
                    xs_i = np.swapaxes(xs_i, 0, 1)
                    xs_i = np.swapaxes(xs_i, 2, 3)

                    ys_i = ys[ i : i + batchsize ]

                if   yield_mc_info == 0:    yield (list(xs_i), ys_i)
                elif yield_mc_info == 1:    yield (list(xs_i), ys_i) + ({ key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() }
                else:   raise ValueError("Wrong argument for yield_mc_info (0/1/2)")
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def getNumEvents(files):
    if isinstance(files, list): pass
    elif isinstance(files, str): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y,files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['EventNumber'].shape[0]
        f.close()
    return counter

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
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['IsMC']  # event ID (0: Data, 1: MC)
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')
    return train_y

def get_events(args, files, model, fOUT):
    try:
        file_ending =  os.path.splitext(fOUT)[1]
        if file_ending == '.p':
            EVENT_INFO = pickle.load(open(fOUT, "rb"))
            write_dict_to_hdf5_file(data=EVENT_INFO, file=(os.path.splitext(fOUT)[0]+'.hdf5'))
        elif file_ending == '.hdf5':
            EVENT_INFO = read_hdf5_file_to_dict(fOUT)
        else:
            raise ValueError('file ending should be .p/.hdf5 but is %s'%(file_ending))
        if events > EVENT_INFO.values()[0].shape[0]: raise IOError
    except IOError:
    #     events_per_batch = 50
    #     if model == None:
    #         raise SystemError('model not found and not events file found')
    #     if events % events_per_batch != 0:
    #         raise ValueError('choose event number in multiples of %f events'%(events_per_batch))
    #
    #     iterations = round_down(args.events, events_per_batch) / events_per_batch
    #     gen = generate_batches_from_files(files, events_per_batch, wires=args.wires, class_type=args.var_targets, f_size=None, yield_mc_info=1)
    #
    #     for i in xrange(iterations):
    #         print i*events_per_batch, ' of ', iterations*events_per_batch
    #         EVENT_INFO_temp = predict_events(model, gen)
    #         if i == 0: EVENT_INFO = EVENT_INFO_temp
    #         else:
    #             for key in EVENT_INFO:
    #                 EVENT_INFO[key] = np.concatenate((EVENT_INFO[key], EVENT_INFO_temp[key]))
    #     # For now, only class probabilities. For final class predictions, do y_classes = y_prob.argmax(axis=-1)
    #     EVENT_INFO['DNNPredClass'] = EVENT_INFO['DNNPred'].argmax(axis=-1)
    #     EVENT_INFO['DNNTrueClass'] = EVENT_INFO['DNNTrue'].argmax(axis=-1)
    #     EVENT_INFO['DNNPredTrueClass'] = EVENT_INFO['DNNPred'][:, 1]
    #     write_dict_to_hdf5_file(data=EVENT_INFO, file=fOUT)
        print ('Check get_events function')
    return EVENT_INFO


def write_dict_to_hdf5_file(data, file, keys_to_write=['all']):
    if not isinstance(data, dict) or not isinstance(file, basestring):
        raise TypeError('passed data/file need to be dict/str. Passed type are: %s/%s'%(type(data),type(file)))
    if 'all' in keys_to_write:
        keys_to_write = data.keys()

    fOUT = h5py.File(file, "w")
    for key in keys_to_write:
        # print 'writing', key
        if key not in data.keys():
            # print keys_to_write, '\n', data.keys()
            raise ValueError('%s not in dict!'%(str(key)))
        fOUT.create_dataset(key, data=np.asarray(data[key]), dtype=np.float32)
    fOUT.close()
    return

def round_down(num, divisor):
    return num - (num%divisor)





# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print (' >> Interrupted << ')

    print ('===================================== Program finished ==============================')

