import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import keras as ks
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
import os
import h5py
from gan import build_generator, build_discriminator, plot_images, make_trainable, get_session


folderRun = '/home/vault/capm/mppi060h/MCDataDiscriminator/TrainingRuns/181106-1720/'
model_file = folderRun + 'models/model-000.hdf5'
weights_file = folderRun + 'models/weights-080.hdf5'
folderInput = '/home/vault/capm/mppi060h/MCDataDiscriminator/Data/mixed_WFs_S5_Th228_P2/'


folderOUT = folderRun + 'output_training/'
folderIN = '/home/vault/capm/mppi060h/MCDataDiscriminator/Data/mixed_WFs_S5_Th228_P2/'
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

log_dir="."


def main():
    files = getFiles()
    executeCNN(files)




def getFiles():
    files = [ os.path.join(folderIN,f) for f in os.listdir(folderIN) ]
    return files


def executeCNN(files):
    
    model = ks.models.load_model(model_file)
    model.load_weights(weights_file)

    print ('Validate events')
    print (model)

    global folderRun
    folderRun += "0validation/test/"
    os.system("mkdir -p -m 770 %s " % (folderRun))
    print(folderRun)

    gen = generate_batches_from_files(files, batchsize=batchsize, wires=wires, class_type=var_targets, yield_mc_info=0)
    for i in range(4000*18):
        wfs, y = next(gen)
        if y[0][0] == 0.0:
            MC_wfs = wfs
            MC_y = y
        else:
            Data_wfs = wfs
            Data_y = y
    GAN_generator(MC_wfs, MC_y, Data_wfs, Data_y, model)




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
            random.shuffle(lst)

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
















def GAN_generator(MC_wfs, MC_y, Data_wfs, Data_y, model):

    data = Data_wfs
    X_train = data 
    X_test = data
    # --------------------------------------------------
    # Set up generator, discriminator and GAN (stacked generator + discriminator)
    # Feel free to modify eg. :
    # - the provided models (see gan.py)
    # - the learning rate
    # - the batchsize
    # --------------------------------------------------

    # Set up generator
    print('\nGenerator')
    latent_dim = 100
    generator = build_generator(latent_dim)
    print(generator.summary())

    # Set up discriminator
    print('\nDiscriminator')
    discriminator = model
    print(discriminator.summary())


    # Set up GAN by stacking the discriminator on top of the generator
    print('\nGenerative Adversarial Network')
    gan_input = Input(shape=[latent_dim])
    gan_output = discriminator(generator(gan_input))
    GAN = Model(gan_input, gan_output)
    print(GAN.summary())
    g_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
    make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
    GAN.compile(loss='binary_crossentropy', optimizer=g_opt)
    # Compile saves the trainable status of the model --> After the model is compiled, updating using make_trainable will have no effect

    # --------------------------------------------------
    # Pretrain the discriminator:
    # --------------------------------------------------

    # - Create a dataset of 10000 real train images and 10000 fake images.
    ntrain = 10000
    no = np.random.choice(60000, size=ntrain, replace='False')
    real_train = X_train[no,:,:,:]   # sample real images from training set
    noise_gen = MC_wfs
    generated_images = generator.predict(noise_gen)  # generate fake images with untrained generator
    print(generated_images.shape)
    X = np.concatenate((real_train, generated_images))
    y = np.zeros([2*ntrain, 2])   # class vector: one-hot encoding
    y[:ntrain, 1] = 1             # class 1 for real images
    y[ntrain:, 0] = 1             # class 0 for generated images

    # - Train the discriminator for 1 epoch on this dataset.
    discriminator.fit(X,y, epochs=1, batch_size=128)

    # - Create a dataset of 5000 real test images and 5000 fake images.
    no = np.random.choice(10000, size=ntrain//2, replace='False')
    real_test = X_test[no,:,:,:]   # sample real images from test set
    noise_gen = MC_wfs//2
    generated_images = generator.predict(noise_gen)    # generate fake images with untrained generator
    Xt = np.concatenate((real_test, generated_images))
    yt = np.zeros([ntrain, 2])   # class vector: one-hot encoding    
    yt[:ntrain//2, 1] = 1         # class 1 for real images
    yt[ntrain//2:, 0] = 1         # class 0 for generated images

    # - Evaluate the test accuracy of your network.
    pretrain_loss, pretrain_acc = discriminator.evaluate(Xt, yt, verbose=0, batch_size=128)
    print('Test accuracy: %04f' % pretrain_acc)

    # loss vector
    losses = {"d":[], "g":[]}
    discriminator_acc = []

    # main training loop
    

    train_for_n(MC_wfs)

    # - Plot the loss of discriminator and generator as function of iterations
    plt.figure(figsize=(10,8))
    plt.semilogy(losses["d"], label='discriminitive loss')
    plt.semilogy(losses["g"], label='generative loss')
    plt.legend()
    plt.savefig(log_dir + '/loss.png')

    # - Plot the accuracy of the discriminator as function of iterations
    plt.figure(figsize=(10,8))
    plt.semilogy(discriminator_acc, label='discriminator accuracy')
    plt.legend()
    plt.savefig(log_dir + '/discriminator_acc.png')

def train_for_n(MC_wfs):
        
    for epoch in range(epochs):
            
        # Plot some fake images
        noise = MC_wfs
        generated_images = generator.predict(noise)
        plot_images(generated_images, fname=log_dir + '/generated_images_' + str(epoch))
        iterations_per_epoch = 60000//batch_size    # number of training steps per epoch
        perm = np.random.choice(60000, size=60000, replace='False')
            
        for i in range(iterations_per_epoch):
                
            # Create a mini-batch of data (X: real images + fake images, y: corresponding class vectors)
            image_batch = X_train[perm[i*batch_size:(i+1)*batch_size],:,:,:]    # real images   
            noise_gen = MC_wfs
            generated_images = generator.predict(noise_gen)                     # generated images
            X = np.concatenate((image_batch, generated_images))
            y = np.zeros([2*batch_size,2])   # class vector
            y[0:batch_size,1] = 1
            y[batch_size:,0] = 1
                
            # Train the discriminator on the mini-batch
            d_loss, d_acc  = discriminator.train_on_batch(X,y)
            losses["d"].append(d_loss)
            discriminator_acc.append(d_acc)

            # Create a mini-batch of data (X: noise, y: class vectors pretending that these produce real images)
            noise_tr = np.random.uniform(0.,1.,size=[batch_size,latent_dim])
            y2 = np.zeros([batch_size,2])
            y2[:,1] = 1

            # Train the generator part of the GAN on the mini-batch
            g_loss = GAN.train_on_batch(noise_tr, y2)
            losses["g"].append(g_loss)











# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print (' >> Interrupted << ')

    print ('===================================== Program finished ==============================') 








