#!/usr/bin/env python

import sys
import os
import stat
import cPickle as pickle

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def parseInput():
    """
        Parses the user input for running the CNN.
        There are three available input modes:
        1) Parse the train/test filepaths directly, if you only have a single file for training/testing
        2) Parse a .list file with arg -l that contains the paths to all train/test files, if the whole dataset is split over multiple files
        3) Parse a .list file with arg -m, if you need multiple input files for a single (!) batch during training.
           This is needed, if e.g. the images for a double input model are coming from different .h5 files.
           An example would be a double input model with two inputs: a loose timecut input (e.g. yzt-x) and a tight timecut input (also yzt-x).
           The important thing is that both files contain the same events, just with different cuts/projections!
           Another usecase would be a double input xyz-t + xyz-c model.
        The output (train_files, test_files) is structured as follows:
        1) train/test: [ ( [train/test_filepath]  , n_rows) ]. The outmost list has len 1 as well as the list in the tuple.
        2) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has arbitrary length (depends on number of files), but len 1 for the list in the tuple.
        3) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has len 1, but the list inside the tuple has arbitrary length.
           A combination of 2) + 3) (multiple input files for each batch from 3) AND all events split over multiple files) is not yet supported.
        :param bool use_scratch_ssd: specifies if the input files should be copied to the node-local SSD scratch space.
        :return: list(([train_filepaths], train_filesize)) train_files: list of tuples that contains the list(trainfiles) and their number of rows.
        :return: list(([test_filepaths], test_filesize)) test_files: list of tuples that contains the list(testfiles) and their number of rows.
    """

    import argparse

    parser = argparse.ArgumentParser(description='E.g. < python run_cnn.py ..... > \n'
                                                 'Script that runs a DNN.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-o', '--out', dest='folderOUT', type=str, default='Dummy', help='folderOUT Path')
    parser.add_argument('-i', '--in', dest='folderIN', type=str, default='/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/', help='folderIN Path')
    parser.add_argument('-r', '--runs', dest='folderRUNS', type=str, default='/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/', help='folderRUNS Path')
    parser.add_argument('-m', '--model', dest='folderMODEL', type=str, default='Dummy', help='folderMODEL Path')
    parser.add_argument('-t', '--targets', type=str, dest='var_targets', default='binary_bb_gamma', help='Targets to train the network against')
    parser.add_argument('-a', '--arch', type=str, dest='cnn_arch', default='DCNN', choices=['DCNN', 'ResNet', 'Inception'], help='Choose network architecture')
    parser.add_argument('-g', '--gpu', type=int, dest='num_gpu', default=1, choices=[1, 2, 3, 4], help='Number of GPUs')
    parser.add_argument('-e', '--epoch', type=int, dest='num_epoch', default=1, help='nb Epochs')
    parser.add_argument('-b', '--batch', type=int, dest='batchsize', default=16, help='Batch Size')
    parser.add_argument('-w', '--weights', dest='num_weights', type=int, default=0, help='Load weights from Epoch')
    parser.add_argument('-s', '--source', dest='sources', default=['mixed'], nargs="*",
                        choices=['mixed', 'bb0n', 'bb0nE', 'bb2n', 'gamma', 'Th228', 'Th232', 'U238', 'Xe137', 'Co60', 'Ra226'], help='sources for training/validation')
    parser.add_argument('-p', '--position', dest='position', default=['AllVessel'], nargs='*', choices=['Uni', 'S2', 'S5', 'S8', 'AllVessel'], help='source position')
    parser.add_argument('-wp', '--wires', type=str, dest='wires', default='U', choices=['U', 'V', 'UV', 'U+V'], help='select wire planes')
    parser.add_argument('-v', '--valid', dest='mode', default='train', choices=['train', 'mc', 'data'], help='mode of operation (train/eval (mc/data))')
    parser.add_argument('-l', '--log', type=str, dest='log', default='', nargs='*', help='Specify settings used for training to distinguish between runs')
    parser.add_argument('--tb', dest='tb_logger', action='store_true', help='activate tensorboard logger')
    parser.add_argument('-ev', '--events', dest='events', default=2000, type=int, help='number of validation events')
    parser.add_argument('--phase', dest='phase', default='2', choices=['1', '2'], help='EXO Phase (1/2)')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--test', dest='test', action='store_true', help='Only reduced data')
    parser.add_argument('--new', dest='new', action='store_true', help='Process new validation events')
    args, unknown = parser.parse_known_args()

    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemError

    folderIN = {}

    if args.log != '':
        print '>>>>>>>>>>>> Settings for this runs <<<<<<<<<<<<'
        print ' '.join(args.log)
        print '>>>>>>>>>>>> <<<<<<<<<<<>>>>>>>>>>> <<<<<<<<<<<<'

    if args.mode == 'data': mode = 'Data'
    elif args.mode == 'train' or args.mode == 'mc': mode = 'MC'
    else: raise ValueError('check mode!')

    args.endings = {}
    for source in args.sources:
        for pos in args.position:
            args.endings[source+pos+mode] = source + '_WFs_' + pos + '_' + mode + '_P' + args.phase

    endings_to_pop = []
    for ending in args.endings:
        folderIN[ending] = os.path.join(os.path.join(os.path.join(args.folderIN,''), args.endings[ending]),'')
        try:
            if not os.path.isdir(folderIN[ending]): raise OSError
            print 'Input  Folder:\t\t', folderIN[ending]
        except OSError:
            endings_to_pop.append(ending)
    for ending in endings_to_pop:
        args.endings.pop(ending)

    args.folderIN = folderIN
    args.folderMODEL = os.path.join(os.path.join(os.path.join(args.folderRUNS,''),args.folderMODEL),'')
    if args.mode == 'train': args.folderOUT = os.path.join(os.path.join(args.folderRUNS, args.folderOUT), '')
    elif args.mode in ['mc', 'data']: args.folderOUT = args.folderMODEL
    else: raise ValueError('wrong mode chosen: %s'%(args.mode))

    adjustPermissions(args.folderOUT)

    if args.resume == True or args.mode != 'train':
        if type(args.num_weights) == int: args.num_weights = str(args.num_weights).zfill(3)
    else:
        args.num_weights = 0
    if not os.path.exists(args.folderOUT+'models'): os.makedirs(args.folderOUT+'models')

    if args.mode == 'data':
        args.var_targets = None

    print 'Output Folder:\t\t'  , args.folderOUT
    if args.resume: print 'Model Folder:\t\t', args.folderMODEL
    print 'Number of GPU:\t\t', args.num_gpu
    print 'Load Epoch:\t\t', args.num_weights
    print 'Number of Epoch:\t', args.num_epoch
    print 'BatchSize:\t\t', args.batchsize, '\n'

    if args.mode == 'train':
        with open(args.folderOUT + 'log.txt', 'w') as f_out:
            args_dict = vars(args)
            for key in args_dict.keys():
                f_out.write(str(key) + '\t' + str(args_dict[key]) + '\n')

    return args

def splitFiles(args, mode, frac_train, frac_val):
    import cPickle as pickle
    files = {}
    if mode == 'train':
        if args.resume:
            os.system("cp %s %s" % (args.folderMODEL + "splitted_files.p", args.folderOUT + "splitted_files.p"))
            print 'load splitted files from %s' % (args.folderMODEL + "splitted_files.p")
            return pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
        else:
            import random
            splitted_files= {'train': {}, 'val': {}, 'test': {}}
            print "\tSource\t\tTotal\tTrain\tValid\tTest"
            for ending in args.endings:
                if (frac_train[ending] + frac_val[ending]) > 1.0 : raise ValueError('check file fractions!')
                files[ending] = [os.path.join(args.folderIN[ending], f) for f in os.listdir(args.folderIN[ending]) if
                                 os.path.isfile(os.path.join(args.folderIN[ending], f))]
                # random.shuffle(files[ending]) #TODO no file shuffling at this moment
                num_train = int(round(len(files[ending]) * frac_train[ending]))
                num_val = int(round(len(files[ending]) * frac_val[ending]))
                if not args.test:
                    splitted_files['train'][ending] = files[ending][0 : num_train]
                    splitted_files['val'][ending]   = files[ending][num_train : num_train + num_val]
                    splitted_files['test'][ending]  = files[ending][num_train + num_val : ]
                else:
                    splitted_files['val'][ending]   = files[ending][0:1]
                    splitted_files['test'][ending]  = files[ending][1:2]
                    splitted_files['train'][ending] = files[ending][2:3]
                print "%s\t\t%i\t%i\t%i\t%i" % (ending, len(files[ending]), len(splitted_files['train'][ending]), len(splitted_files['val'][ending]), len(splitted_files['test'][ending]))
            pickle.dump(splitted_files, open(args.folderOUT + "splitted_files.p", "wb"))
            return splitted_files
    elif mode == 'mc':
        files_training = pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
        for ending in args.endings:
            if ending in files_training['val'].keys() or ending in files_training['test'].keys():
                files[ending] = files_training['val'][ending] + files_training['test'][ending]
            else:
                files[ending] = [os.path.join(args.folderIN[ending], f) for f in os.listdir(args.folderIN[ending]) if
                                 os.path.isfile(os.path.join(args.folderIN[ending], f))]
        print 'Input  File:\t\t', (args.folderOUT + "splitted_files.p")
        return files
    elif mode == 'data':
        for ending in args.endings:
            files[ending] = [os.path.join(args.folderIN[ending], f) for f in os.listdir(args.folderIN[ending]) if
                             os.path.isfile(os.path.join(args.folderIN[ending], f))]
        return files
    else:
        raise ValueError('mode is not valid')

def adjustPermissions(path):
    # set this folder to read/writeable/exec
    try:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
    except OSError:
        # TODO could copy and replace non-changeable files and apply chmod on new files
        pass

    # step through all the files/folders and change permissions
    for file in os.listdir(path):
        filePath = os.path.join(path, file)

        # if it is a directory, doe recursive call
        if os.path.isdir(filePath):
            adjustPermissions(filePath)
        # for files merely call chmod
        else:
            try:
                # set this file to read/writeable/exec
                os.chmod(filePath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
            except OSError:
                # TODO could copy and replace non-changeable files and apply chmod on new files
                pass

