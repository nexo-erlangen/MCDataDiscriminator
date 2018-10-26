#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import keras as ks
import os
import numpy as np
from keras import backend as K

from input_utilities import *
from generator import *
from plot_scripts import plot_input_plots
from plot_scripts import plot_traininghistory
from plot_scripts import plot_validation

class TensorBoardWrapper(ks.callbacks.TensorBoard):
    """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
     Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
     Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.

     This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
     that the whole validation data is put into a single array by using the generator.
     Then, the single array is used in the validation steps. This workaround is experimental!"""
    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in xrange(self.nb_steps):
            ib, tb = next(self.batch_gen)
            ib = np.asarray(ib)
            if imgs is None and tags is None:
                imgs = np.zeros(((ib.shape[0],) + (self.nb_steps * ib.shape[1],) + ib.shape[2:]), dtype=ib.dtype)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=tb.dtype)
            imgs[ : , s * ib.shape[1]:(s + 1) * ib.shape[1]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs[0], imgs[1], tags, np.ones(imgs[0].shape[0]), 0.0]
        # self.validation_data = [list(imgs), tags, np.ones(imgs[0].shape[0]), 0.0]
        print len(self.validation_data)

        print self.model.inputs[0].shape

        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
            print 'learn phase', K.learning_phase()
            tensors += [K.learning_phase()]

        print len(tensors)

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class BatchLevelPerformanceLogger(ks.callbacks.Callback):
    # Gibt loss aus über alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display, skipBatchesVal, steps_per_epoch, args, genVal):
        ks.callbacks.Callback.__init__(self)
        self.Valseen = 0
        self.averageLoss = 0.0
        self.averageAcc = 0.0
        self.averageValLoss = 0.0
        self.averageValAcc = 0.0
        self.steps_per_epoch = steps_per_epoch
        self.steps = steps_per_epoch // display
        self.skipBatchesVal = skipBatchesVal
        self.args = args
        self.seen = int(self.args.num_weights) * self.steps_per_epoch
        self.logfile_train_fname = self.args.folderOUT + 'log_train.txt'
        self.logfile_train = None
        self.genVal = genVal

    def on_train_begin(self, logs={}):
        if self.args.resume:
            os.system("cp %s %s" % (self.args.folderMODEL + 'log_train.txt', self.logfile_train_fname))
        return

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageAcc += logs.get('acc')

        if self.seen % self.skipBatchesVal == 0:
            self.Valseen += 1
            valLoss, valAcc = tuple(self.model.evaluate_generator(self.genVal, steps=1))
            self.averageValLoss += valLoss
            self.averageValAcc += valAcc

        if self.seen % self.steps == 0:
            averaged_loss = self.averageLoss / self.steps
            averaged_acc = self.averageAcc / self.steps
            averaged_ValLoss = self.averageValLoss / self.Valseen if self.Valseen > 0 else 0.0
            averaged_ValAcc = self.averageValAcc / self.Valseen if self.Valseen > 0 else 0.0

            print
            print 'average loss/acc train:', averaged_loss, averaged_acc
            print 'average loss/acc test:', averaged_ValLoss, averaged_ValAcc

            batchnumber_float = (self.seen - self.steps / 2.) / float(self.steps_per_epoch) # + self.epoch - 1  # start from zero
            self.loglist.append('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(self.seen, batchnumber_float, averaged_loss, averaged_acc, averaged_ValLoss, averaged_ValAcc))
            self.averageLoss = 0
            self.averageAcc = 0
            self.averageValLoss = 0
            self.averageValAcc = 0
            self.Valseen = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.loglist = []

    def on_epoch_end(self, epoch, logs={}):
        self.logfile_train = open(self.logfile_train_fname, 'a+')
        if os.stat(self.logfile_train_fname).st_size == 0: self.logfile_train.write("#Batch\tBatch_float\tLoss\tAcc\tValLoss\tValAcc")

        for batch_statistics in self.loglist: # only write finished epochs to the .txt
            self.logfile_train.write(batch_statistics)

        self.logfile_train.flush()
        os.fsync(self.logfile_train.fileno())
        self.logfile_train.close()
        try:
            plot_validation.plot_learning_curve(self.args.folderOUT, np.loadtxt(self.logfile_train_fname, unpack=True))
        except: print 'plotting learning curve not successfull. Skipping'


class EpochLevelPerformanceLogger(ks.callbacks.Callback):
    def __init__(self, args, files, var_targets):
        ks.callbacks.Callback.__init__(self)
        self.validation_data = None
        self.args = args
        self.files = files
        self.eventsVal = min([getNumEvents(self.files), 2000])
        self.eventsPerBatch = 50
        self.iterationsVal = round_down(self.eventsVal, self.eventsPerBatch) / self.eventsPerBatch
        self.genVal = generate_batches_from_files(self.files, batchsize=self.eventsPerBatch, class_type=var_targets, yield_mc_info=1)

    def on_train_begin(self, logs={}):
        self.losses = []
        if self.args.resume:
            os.system("cp %s %s" % (self.args.folderMODEL + "save.p", self.args.folderOUT + "save.p"))
        else:
            pickle.dump({}, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        Y_PRED, Y_TRUE, EVENT_INFO = [], [], []
        for i in xrange(self.val_iterations):
            Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp = predict_events(self.model, self.genVal)
            Y_PRED.extend(Y_PRED_temp)
            Y_TRUE.extend(Y_TRUE_temp)
            EVENT_INFO.extend(EVENT_INFO_temp)
        # Eval_dict = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': np.asarray(EVENT_INFO)}
        # obs = plot.make_plots(self.args.folderOUT, dataIn=dataIn, epoch=str(epoch), sources='th', position='S5')
        self.dict_out = pickle.load(open(self.args.folderOUT + "save.p", "rb"))
        self.dict_out[epoch] = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': np.asarray(EVENT_INFO),
                                'loss': logs['loss'], 'acc': logs['acc'],
                                'val_loss': logs['val_loss'], 'val_acc': logs['val_acc']}
        pickle.dump(self.dict_out, open(self.args.folderOUT + "save.p", "wb"))
        on_epoch_end_plots(folderOUT=self.args.folderOUT, epoch=epoch, data=self.dict_out[epoch])
        # print logs.keys()


        # plot_train_and_test_statistics(modelname)
        # plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels, modelname,
        #                              epoch[0], file_no, str_ident)
        # plot_traininghistory(args)

        return