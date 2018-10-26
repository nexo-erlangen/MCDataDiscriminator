#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
import cPickle as pickle

##################################################################################################

folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181002-0836/0validation/ShapeAgreement-014/'
files = {}
files['1'] = '../Th228-mc-S5-014-UV/events_014_Th228-mc-S5-UV.p'
files['2'] = '../Th228-data-S5-014-UV/events_014_Th228-data-S5-UV.p'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement/'
# files = {}
# files['1'] = '../Th228-mc-S5-023-U/events_023_Th228-mc-S5-U.p'
# files['2'] = '../Th228-data-S5-023-U/events_023_Th228-data-S5-U.p'

discriminator = 'signal-likeness'

def main():
    data = {}
    for key, model in files.items():
        data[key] = pickle.load(open(folderRUNS + files[key], "rb"))

    mask1 = (data['1']['CCIsSS'] == 1)
    mask2 = (data['2']['CCIsSS'] == 1)

    rad1 = np.sqrt(data['1']['CCPosX'][:, 0] * data['1']['CCPosX'][:, 0] + data['1']['CCPosY'][:, 0] * data['1']['CCPosY'][:, 0])
    rad2 = np.sqrt(data['2']['CCPosX'][:, 0] * data['2']['CCPosX'][:, 0] + data['2']['CCPosY'][:, 0] * data['2']['CCPosY'][:, 0])

    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], rad1[mask1],
                     data['2']['DNNPredTrueClass'][mask2], rad2[mask2],
                     [0.0, 1.0], [0, 180], discriminator, 'R', 'MC+Data', '%s_vs_R_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosX'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosX'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'X', 'MC+Data', '%s_vs_X_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosY'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosY'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'Y', 'MC+Data', '%s_vs_Y_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosZ'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosZ'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'Z', 'MC+Data', '%s_vs_Z_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1],
                     data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2],
                     [0.0, 1.0], [1000, 3000], discriminator, 'Energy', 'MC+Data', '%s_vs_Energy_SS.pdf'%(discriminator))


    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], rad1[mask1],
                          data['2']['DNNPredTrueClass'][mask2], rad2[mask2],
                          [0.0, 1.0], [0, 180], discriminator, 'R', 'MC+Data', '%s_vs_R_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosX'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosX'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'X', 'MC+Data', '%s_vs_X_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosY'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosY'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'Y', 'MC+Data', '%s_vs_Y_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosZ'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosZ'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'Z', 'MC+Data', '%s_vs_Z_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1],
                          data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2],
                          [0.0, 1.0], [1000, 2500], discriminator, 'Energy', 'MC+Data', '%s_vs_Energy_SS_norm.pdf'%(discriminator))

    mask_1y = data['1']['DNNTrueClass'] == 0
    mask_2y = data['2']['DNNTrueClass'] == 0

    for i in xrange(3):
        if i == 0:
            continue
            mask1 = mask2 = True
            title = 'SS+MS'
        elif i == 1:
            mask1 = (data['1']['CCIsSS'] == 1)
            mask2 = (data['2']['CCIsSS'] == 1)
            title = 'SS'
        elif i == 2:
            # continue
            mask1 = (data['1']['CCIsSS'] == 0)
            mask2 = (data['2']['CCIsSS'] == 0)
            title = 'MS'
        else:
            raise ValueError('check loop')

        rad1 = np.sqrt(data['1']['CCPosX'][:, 0] * data['1']['CCPosX'][:, 0] + data['1']['CCPosY'][:, 0] * data['1']['CCPosY'][:, 0])
        rad2 = np.sqrt(data['2']['CCPosX'][:, 0] * data['2']['CCPosX'][:, 0] + data['2']['CCPosY'][:, 0] * data['2']['CCPosY'][:, 0])

        kwargs = {
            'range': (1000, 3000),
            'bins': 50,
            'density': False
        }

        make_shape_agreement_plot(np.sum(data['1']['CCPurityCorrectedEnergy'][mask_1y & mask1], axis=1),
                                  np.sum(data['2']['CCPurityCorrectedEnergy'][mask_2y & mask2], axis=1), title, 'energy', **kwargs)

        kwargs = {
            'range': (0, 1),
            'bins': 50,
            'density': False
        }

        e_limit = 2000.
        maskE1 = np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1) > e_limit
        maskE2 = np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1) > e_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskE1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskE2] , title, '%s (E gr %d)' % (discriminator, e_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskE1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskE2], title, '%s (E le %d)' % (discriminator, e_limit), **kwargs)


        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1], data['2']['DNNPredTrueClass'][mask_2y & mask2], title, discriminator, **kwargs)

        rad_limit = 160.
        maskR1 = rad1 < rad_limit
        maskR2 = rad2 < rad_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskR1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskR2] , title, '%s (R le %d)'%(discriminator, rad_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskR1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskR2], title, '%s (R gr %d)'%(discriminator, rad_limit), **kwargs)

        z_limit = 20.
        maskZ1 = np.abs(data['1']['CCPosZ'][:, 0]) > z_limit
        maskZ2 = np.abs(data['2']['CCPosZ'][:, 0]) > z_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskZ1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskZ2] , title, '%s (Z gr %d)'%(discriminator, z_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskZ1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskZ2], title, '%s (Z le %d)'%(discriminator, z_limit), **kwargs)


        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskZ1 & maskR1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskZ2 & maskR2] , title, '%s (Z gr %d + R le %d)'%(discriminator, z_limit, rad_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~(maskZ1 & maskR1)], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~(maskZ2 & maskR2)], title, '%s not(Z gr %d + R le %d)'%(discriminator, z_limit, rad_limit), **kwargs)


        # kwargs = {
        #     'range': (-180, 180),
        #     'bins': 50,
        #     'density': False
        # }
        # make_shape_agreement_plot(data['1'], data['2'], data['1']['CCPosZ'][:,0], data['2']['CCPosZ'][:,0], 'Z', **kwargs)
        #
        # kwargs = {
        #     'range': (0, 175),
        #     'bins': 50,
        #     'density': False
        # }
        # make_shape_agreement_plot(data['1'], data['2'], rad1, rad2, 'radius', **kwargs)




def make_shape_agreement_plot(mc, data, title, name, **kwargs):
    hist_1y, bin_edges = np.histogram(mc, **kwargs)
    hist_2y, bin_edges = np.histogram(data, **kwargs)

    hist_2y_err = np.sqrt(hist_2y)

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    nevents1 = float(sum(hist_1y))
    nevents2 = float(sum(hist_2y))
    binwidth = (bin_edges[1] - bin_edges[0])
    hist_1y = hist_1y / nevents1 / binwidth
    hist_2y = hist_2y / nevents2 / binwidth
    hist_2y_err = hist_2y_err / nevents2 / binwidth

    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex = ax1)
    ax1.step(bin_centres, hist_1y, where='mid', color='blue', label='MC (%s)'%(title))
    ax1.errorbar(bin_centres, hist_2y, hist_2y_err, color='k', fmt='.', label='Th228 (S5)')
    ax2.axhline(y=0., c='k')
    ax2.errorbar(bin_centres, (hist_2y-hist_1y)/hist_1y, hist_2y_err/hist_1y, color='k', fmt='.', label='Th228 (S5)')
    # ax2.scatter(bin_centres, (hist_2y-hist_1y)/hist_2y_err, color='k')
    ax2.set_xlabel(name)
    ax2.set_ylabel('(data-MC)/MC')
    ax1.legend(loc='upper center')
    # ax1.set_title(title)
    ax1.set_xlim(kwargs['range'])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(-0.7, 0.7)
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)
    f.savefig(folderRUNS + name+ '_' + title + '.pdf', bbox_inches='tight')
    plt.close()

#TODO Check difference plot ax3. Looks strange
def plot_hist2_multi_norm(E_x1, E_y1, E_x2, E_y2, range_x, range_y, name_x, name_y, name_title, fOUT):
    pos_bins = 25

    hist1D_Z1, bin_edges = np.histogram(E_y1, range=range_y, bins=pos_bins, normed=True)
    hist1D_Z2, bin_edges = np.histogram(E_y2, range=range_y, bins=pos_bins, normed=True)

    weights_Z1 = np.asarray([1. / hist1D_Z1[np.argmax(bin_edges >= p) - 1]
                             if p >= bin_edges[0] and p < bin_edges[-1] else 0.0 for p in E_y1])
    weights_Z2 = np.asarray([1. / hist1D_Z2[np.argmax(bin_edges >= p) - 1]
                             if p >= bin_edges[0] and p < bin_edges[-1] else 0.0 for p in E_y2])

    hist2D_Z1, xbins, ybins = np.histogram2d(E_x1, E_y1, weights=weights_Z1, range=[range_x, range_y],
                                            bins=pos_bins, normed=True)
    hist2D_Z2, xbins, ybins = np.histogram2d(E_x2, E_y2, weights=weights_Z2, range=[range_x, range_y],
                                            bins=pos_bins, normed=True)

    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = "auto"

    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    max12 = 100. * np.max([hist2D_Z1, hist2D_Z2])
    h2 = ax2.imshow(100.*hist2D_Z1.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                   aspect=aspect, norm=colors.Normalize(vmax=max12))
    f.colorbar(h2, ax=ax2, shrink=0.8)
    h1 = ax1.imshow(100.*hist2D_Z2.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=colors.Normalize(vmax=max12))
    f.colorbar(h1, ax=ax1, shrink=0.8)

    hist2D_Z3 = (hist2D_Z2 - hist1D_Z1)
    max3 = 100.*np.max(np.abs(hist2D_Z3))
    h3 = ax3.imshow(100.*hist2D_Z3.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('RdBu_r'), origin='lower',
                    aspect=aspect, norm=colors.Normalize(vmin=-max3, vmax=max3))
    f.colorbar(h3, ax=ax3, shrink=0.8)

    ax3.set_xlabel(name_x)
    ax1.set_ylabel('%s (Data)'%(name_y))
    ax2.set_ylabel('%s (MC)'%(name_y))
    ax3.set_ylabel('%s (Data-MC)'%(name_y))
    ax1.set_xlim(range_x)
    ax1.set_ylim(range_y)
    ax2.set_ylim(range_y)
    ax3.set_ylim(range_y)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    f.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.close()


def plot_hist2_multi(E_x1, E_y1, E_x2, E_y2, range_x, range_y, name_x, name_y, name_title, fOUT):
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    numEv = min((E_x1.shape[0],E_x2.shape[0]))
    # E_y1 = np.abs(E_y1)
    # E_y2 = np.abs(E_y2)

    h2 = ax2.hexbin(E_x1[:numEv], E_y1[:numEv], extent=extent, gridsize=25, linewidths=0.1, norm=colors.Normalize(vmax=120), cmap=plt.get_cmap('viridis'))
    f.colorbar(h2, ax=ax1, shrink=0.6)
    h1 = ax1.hexbin(E_x2[:numEv], E_y2[:numEv], extent=extent, gridsize=25, linewidths=0.1, norm=colors.Normalize(vmax=120), cmap=plt.get_cmap('viridis'))
    f.colorbar(h1, ax=ax2, shrink=0.6)

    # max3 = np.max(h2.get_array() - h1.get_array())
    h3 = ax3.hexbin(E_x2[:numEv], E_y2[:numEv], extent=extent, gridsize=25, linewidths=0.1, vmin=-50, vmax=50, cmap=plt.get_cmap('RdBu_r'))
    h3.set_array(h2.get_array() - h1.get_array())
    f.colorbar(h3, ax=ax3, shrink=0.6)

    ax3.set_xlabel(name_x)
    ax1.set_ylabel('%s (Data)'%(name_y))
    ax2.set_ylabel('%s (MC)'%(name_y))
    ax3.set_ylabel('%s (Data-MC)'%(name_y))
    ax1.set_xlim(range_x)
    ax1.set_ylim(range_y)
    ax2.set_ylim(range_y)
    ax3.set_ylim(range_y)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.subplots_adjust(hspace=.0)
    f.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.close()


# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, range_x, range_y, name_x, name_y, name_title, fOUT):
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    plt.hexbin(E_x, E_y, extent=extent, gridsize=50, mincnt=1, linewidths=0.1, cmap=plt.get_cmap('viridis'))

    plt.title('%s' % (name_title))
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.xlim(xmin=range_x[0], xmax=range_x[1])
    plt.ylim(ymin=range_y[0], ymax=range_y[1])
    plt.grid(True)
    plt.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

if __name__ == '__main__':

    main()

    print '===================================== Program finished =============================='
