#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
import cPickle as pickle

# scatter
def plot_scatter(E_x, E_y, name_x, name_y, name_title, fOUT):
    plt.scatter(E_x, E_y)
    plt.plot([0,1], [0,1], 'k--')
    # plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,1],[0,1]], bins=50, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    # plt.plot([0,1], [0,1], 'k--')
    # im = plt.imshow(hist.T, extent=extent, interpolation='nearest', vmin=0, vmax=1, cmap=plt.get_cmap('viridis'), origin='lower', norm=mpl.colors.Normalize()) # norm=mpl.colors.LogNorm())
    plt.hexbin(E_x, E_y, extent=extent, gridsize=100, mincnt=1, linewidths=0.1, cmap=plt.get_cmap('viridis'))  # norm=mpl.colors.LogNorm())

    #cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    #cbar.set_label('Probability')
    plt.title('%s' % (name_title))
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


##################################################################################################

folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/Dummy/CompareModels/'
files = {}
files['1'] = 'events_023_mixed-Uni-U.p' #0validation-mc/mixed-Uni-023-U/events_023_mixed-Uni.p'
files['2'] = 'events_023_mixed-Uni-V.p' #0validation-mc/mixed-Uni-023-V/events_023_mixed-Uni.p'

data = {}
for key,model in files.items():
    data[key] = pickle.load(open(folderRUNS + files[key], "rb"))
    data[key]['MCRunEventNumber'] = [(RN, EN) for RN, EN in zip(data[key]['MCRunNumber'], data[key]['MCEventNumber'])]

data['mean'] = {}
data['mean']['DNNPred'] = np.add(data['1']['DNNPred'], data['2']['DNNPred'])/2.
data['mean']['DNNPredClass'] = data['mean']['DNNPred'].argmax(axis=-1)
data['mean']['DNNPredTrueClass'] = data['mean']['DNNPred'][:, 1]
data['mean']['DNNTrueClass'] = data['1']['DNNTrue'].argmax(axis=-1)



data['max'] = {}
# test = (2.0*data['1']['DNNPred'][:,0]*data['2']['DNNPred'][:,0])/(data['1']['DNNPred'][:,0]+data['2']['DNNPred'][:,0])
# test = np.sqrt(data['1']['DNNPred'][:,0]*data['2']['DNNPred'][:,0])
# test = np.sqrt(data['1']['DNNPred'][:,0]**2+data['2']['DNNPred'][:,0]**2)/np.sqrt(2.)
test = np.maximum(data['1']['DNNPred'][:,0], data['2']['DNNPred'][:,0])
data['max']['DNNPred'] = np.stack((test,1.-test), axis=1)

# test = np.sqrt(data['1']['DNNPred'][:,1]**2+data['2']['DNNPred'][:,1]**2)/np.sqrt(2.)
# data['max']['DNNPred'] = np.stack((1.-test,test), axis=1)


# print data['max']['DNNPred'].shape
# data['max']['DNNPred'] = np.swapaxes(data['max']['DNNPred'], 0, 1)
# print data['max']['DNNPred'].shape

for i in range(2):
    print data['1']['DNNPred'][i], data['2']['DNNPred'][i], data['max']['DNNPred'][i]


data['max']['DNNPredClass'] = data['max']['DNNPred'].argmax(axis=-1)
data['max']['DNNPredTrueClass'] = data['max']['DNNPred'][:, 1]
data['max']['DNNTrueClass'] = data['1']['DNNTrue'].argmax(axis=-1)

kwargs = {
    'range': (0, 1),
    'bins': 100,
    'density': True
}

hist_1ee, bin_edges = np.histogram(data['1']['DNNPredTrueClass'][data['1']['DNNTrueClass'] == 1], **kwargs)
hist_1y, bin_edges = np.histogram(data['1']['DNNPredTrueClass'][data['1']['DNNTrueClass'] == 0], **kwargs)

hist_2ee, bin_edges = np.histogram(data['2']['DNNPredTrueClass'][data['2']['DNNTrueClass'] == 1], **kwargs)
hist_2y, bin_edges = np.histogram(data['2']['DNNPredTrueClass'][data['2']['DNNTrueClass'] == 0], **kwargs)

hist_mean_ee, bin_edges = np.histogram(data['mean']['DNNPredTrueClass'][data['mean']['DNNTrueClass'] == 1], **kwargs)
hist_mean_y, bin_edges = np.histogram(data['mean']['DNNPredTrueClass'][data['mean']['DNNTrueClass'] == 0], **kwargs)

hist_max_ee, bin_edges = np.histogram(data['max']['DNNPredTrueClass'][data['max']['DNNTrueClass'] == 1], **kwargs)
hist_max_y, bin_edges = np.histogram(data['max']['DNNPredTrueClass'][data['max']['DNNTrueClass'] == 0], **kwargs)

# hist_ee_SS, bin_edges = np.histogram(
#     data['DNNPredTrueClass'][(data['DNNTrueClass'] == 1) & (data['CCIsSS'] == 1)], **kwargs)
# hist_y_SS, bin_edges = np.histogram(
#     data['DNNPredTrueClass'][(data['DNNTrueClass'] == 0) & (data['CCIsSS'] == 1)], **kwargs)
#
# hist_ee_MS, bin_edges = np.histogram(
#     data['DNNPredTrueClass'][(data['DNNTrueClass'] == 1) & (data['CCIsSS'] == 0)], **kwargs)
# hist_y_MS, bin_edges = np.histogram(
#     data['DNNPredTrueClass'][(data['DNNTrueClass'] == 0) & (data['CCIsSS'] == 0)], **kwargs)

bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

# norm_ms = (data['CCIsSS'] == 1).sum()/float(len(data['CCIsSS']))

plt.clf()
plt.step(bin_centres, hist_1y, where='mid', color='firebrick', label='U-gamma')
plt.step(bin_centres, hist_1ee, where='mid', color='blue', label='U-2beta')
plt.step(bin_centres, hist_2y, where='mid', color='firebrick', alpha=0.4, label='V-gamma')
plt.step(bin_centres, hist_2ee, where='mid', color='blue', alpha=0.4, label='V-2beta')
plt.xlabel('threshold')
plt.legend(loc='upper center')
plt.title('SS+MS')
plt.xlim([0, 1])
plt.ylim(ymin=0)
plt.savefig(folderRUNS + 'histogram_vs_threshold-UorV.pdf', bbox_inches='tight')
plt.close()

plt.clf()
plt.step(bin_centres, hist_mean_y, where='mid', color='firebrick', label='<U+V>-gamma')
plt.step(bin_centres, hist_mean_ee, where='mid', color='blue', label='<U+V>-2beta')
plt.step(bin_centres, hist_max_y, where='mid', color='firebrick', alpha=0.4, label='max(U,V)-gamma')
plt.step(bin_centres, hist_max_ee, where='mid', color='blue', alpha=0.4, label='max(U,V)-2beta')
# plt.fill_between(bin_centres, np.zeros(hist_y_SS.shape), hist_y_MS*(1-norm_ms), label='gamma (MS)', step='mid', color='firebrick', alpha=0.3)
# plt.fill_between(bin_centres, np.zeros(hist_ee_SS.shape), hist_ee_MS*(1-norm_ms), label='2beta (MS)', step='mid', color='blue', alpha=0.3)
plt.xlabel('threshold')
plt.legend(loc='upper center')
plt.title('SS+MS')
plt.xlim([0, 1])
plt.ylim(ymin=0)
plt.savefig(folderRUNS + 'histogram_vs_threshold-UandV.pdf', bbox_inches='tight')
plt.close()

from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score

maskSS = data['1']['CCIsSS'] == 1
maskMS = data['1']['CCIsSS'] == 0

data['1']['prc'] = precision_recall_curve(data['1']['DNNTrueClass'], data['1']['DNNPredTrueClass'])
data['2']['prc'] = precision_recall_curve(data['2']['DNNTrueClass'], data['2']['DNNPredTrueClass'])
data['mean']['prc'] = precision_recall_curve(data['mean']['DNNTrueClass'], data['mean']['DNNPredTrueClass'])
data['max']['prc'] = precision_recall_curve(data['max']['DNNTrueClass'], data['max']['DNNPredTrueClass'])

data['1']['roc-SS'] = roc_curve(data['1']['DNNTrueClass'][maskSS], data['1']['DNNPredTrueClass'][maskSS])
data['2']['roc-SS'] = roc_curve(data['2']['DNNTrueClass'][maskSS], data['2']['DNNPredTrueClass'][maskSS])
data['1']['roc_auc-SS'] = roc_auc_score(data['1']['DNNTrueClass'][maskSS], data['1']['DNNPredTrueClass'][maskSS])
data['2']['roc_auc-SS'] = roc_auc_score(data['2']['DNNTrueClass'][maskSS], data['2']['DNNPredTrueClass'][maskSS])
data['1']['roc-MS'] = roc_curve(data['1']['DNNTrueClass'][maskMS], data['1']['DNNPredTrueClass'][maskMS])
data['2']['roc-MS'] = roc_curve(data['2']['DNNTrueClass'][maskMS], data['2']['DNNPredTrueClass'][maskMS])
data['1']['roc_auc-MS'] = roc_auc_score(data['1']['DNNTrueClass'][maskMS], data['1']['DNNPredTrueClass'][maskMS])
data['2']['roc_auc-MS'] = roc_auc_score(data['2']['DNNTrueClass'][maskMS], data['2']['DNNPredTrueClass'][maskMS])
data['1']['roc'] = roc_curve(data['1']['DNNTrueClass'], data['1']['DNNPredTrueClass'])
data['2']['roc'] = roc_curve(data['2']['DNNTrueClass'], data['2']['DNNPredTrueClass'])
data['1']['roc_auc'] = roc_auc_score(data['1']['DNNTrueClass'], data['1']['DNNPredTrueClass'])
data['2']['roc_auc'] = roc_auc_score(data['2']['DNNTrueClass'], data['2']['DNNPredTrueClass'])


data['mean']['roc-SS'] = roc_curve(data['mean']['DNNTrueClass'][maskSS], data['mean']['DNNPredTrueClass'][maskSS])
data['max']['roc-SS'] = roc_curve(data['max']['DNNTrueClass'][maskSS], data['max']['DNNPredTrueClass'][maskSS])
data['mean']['roc_auc-SS'] = roc_auc_score(data['mean']['DNNTrueClass'][maskSS], data['mean']['DNNPredTrueClass'][maskSS])
data['max']['roc_auc-SS'] = roc_auc_score(data['max']['DNNTrueClass'][maskSS], data['max']['DNNPredTrueClass'][maskSS])
data['mean']['roc-MS'] = roc_curve(data['mean']['DNNTrueClass'][maskMS], data['mean']['DNNPredTrueClass'][maskMS])
data['max']['roc-MS'] = roc_curve(data['max']['DNNTrueClass'][maskMS], data['max']['DNNPredTrueClass'][maskMS])
data['mean']['roc_auc-MS'] = roc_auc_score(data['mean']['DNNTrueClass'][maskMS], data['mean']['DNNPredTrueClass'][maskMS])
data['max']['roc_auc-MS'] = roc_auc_score(data['max']['DNNTrueClass'][maskMS], data['max']['DNNPredTrueClass'][maskMS])
data['mean']['roc'] = roc_curve(data['mean']['DNNTrueClass'], data['mean']['DNNPredTrueClass'])
data['max']['roc'] = roc_curve(data['max']['DNNTrueClass'], data['max']['DNNPredTrueClass'])
data['mean']['roc_auc'] = roc_auc_score(data['mean']['DNNTrueClass'], data['mean']['DNNPredTrueClass'])
data['max']['roc_auc'] = roc_auc_score(data['max']['DNNTrueClass'], data['max']['DNNPredTrueClass'])


plt.clf()
plt.plot(data['1']['prc'][1], data['1']['prc'][0], label='U')
plt.plot(data['2']['prc'][1], data['2']['prc'][0], label='V')
plt.plot(data['mean']['prc'][1], data['mean']['prc'][0], label='<U+V>', lw=2)
plt.plot(data['max']['prc'][1], data['max']['prc'][0], label='max(U,V)', lw=2, c='k')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc='best')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(folderRUNS+ 'precision_vs_recall.pdf', bbox_inches='tight')
plt.close()

plt.clf()
plt.plot(data['1']['roc'][0], data['1']['roc'][1], label='U (%.1f%%)' % (data['1']['roc_auc']*100.))
plt.plot(data['2']['roc'][0], data['2']['roc'][1], label='V (%.1f%%)' % (data['2']['roc_auc']*100.))
plt.plot(data['mean']['roc'][0], data['mean']['roc'][1], label='<U+V> (%.1f%%)' % (data['mean']['roc_auc']*100.))
plt.plot(data['max']['roc'][0], data['max']['roc'][1], label='max(U,V) (%.1f%%)' % (data['max']['roc_auc']*100.))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(folderRUNS + 'roc_curve.pdf', bbox_inches='tight')
plt.close()

plt.clf()
plt.plot(data['1']['roc'][0], data['1']['roc'][1], label='U (%.1f%%)' % (data['1']['roc_auc']*100.))
plt.plot(data['2']['roc'][0], data['2']['roc'][1], label='V (%.1f%%)' % (data['2']['roc_auc']*100.))
plt.plot(data['1']['roc-SS'][0], data['1']['roc-SS'][1], label='U SS (%.1f%%)' % (data['1']['roc_auc-SS']*100.))
plt.plot(data['2']['roc-SS'][0], data['2']['roc-SS'][1], label='V SS (%.1f%%)' % (data['2']['roc_auc-SS']*100.))
plt.plot(data['1']['roc-MS'][0], data['1']['roc-MS'][1], label='U MS (%.1f%%)' % (data['1']['roc_auc-MS']*100.))
plt.plot(data['2']['roc-MS'][0], data['2']['roc-MS'][1], label='V MS (%.1f%%)' % (data['2']['roc_auc-MS']*100.))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(folderRUNS + 'roc_U-V_SS-MS_curve.pdf', bbox_inches='tight')
plt.close()

plt.clf()
plt.plot(data['mean']['roc'][0], data['mean']['roc'][1], label='<U+V> (%.1f%%)' % (data['mean']['roc_auc']*100.))
plt.plot(data['max']['roc'][0], data['max']['roc'][1], label='max(U,V) (%.1f%%)' % (data['max']['roc_auc']*100.))
plt.plot(data['mean']['roc-SS'][0], data['mean']['roc-SS'][1], label='<U+V> SS (%.1f%%)' % (data['mean']['roc_auc-SS']*100.))
plt.plot(data['max']['roc-SS'][0], data['max']['roc-SS'][1], label='max(U,V) SS (%.1f%%)' % (data['max']['roc_auc-SS']*100.))
plt.plot(data['mean']['roc-MS'][0], data['mean']['roc-MS'][1], label='<U+V> MS (%.1f%%)' % (data['mean']['roc_auc-MS']*100.))
plt.plot(data['max']['roc-MS'][0], data['max']['roc-MS'][1], label='max(U,V) MS (%.1f%%)' % (data['max']['roc_auc-MS']*100.))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(folderRUNS + 'roc_U+V_SS-MS_curve.pdf', bbox_inches='tight')
plt.close()

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][data['1']['DNNTrueClass'] == 1],
                    data['2']['DNNPredTrueClass'][data['2']['DNNTrueClass'] == 1],
                    'U', 'V', '2beta events', folderRUNS + 'hist2d_SS+MS_signal.pdf')

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][data['1']['DNNTrueClass'] == 0],
                         data['2']['DNNPredTrueClass'][data['2']['DNNTrueClass'] == 0],
                         'U', 'V', 'gamma events', folderRUNS + 'hist2d_SS+MS_background.pdf')

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][(data['1']['DNNTrueClass'] == 1) & (data['1']['CCIsSS'] == 1)],
                    data['2']['DNNPredTrueClass'][(data['2']['DNNTrueClass'] == 1) & (data['1']['CCIsSS'] == 1)],
                    'U', 'V', '2beta events', folderRUNS + 'hist2d_SS_signal.pdf')

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][(data['1']['DNNTrueClass'] == 0) & (data['1']['CCIsSS'] == 1)],
                    data['2']['DNNPredTrueClass'][(data['2']['DNNTrueClass'] == 0) & (data['1']['CCIsSS'] == 1)],
                    'U', 'V', 'gamma events', folderRUNS + 'hist2d_SS_background.pdf')

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][(data['1']['DNNTrueClass'] == 1) & (data['1']['CCIsSS'] == 0)],
                    data['2']['DNNPredTrueClass'][(data['2']['DNNTrueClass'] == 1) & (data['1']['CCIsSS'] == 0)],
                    'U', 'V', '2beta events', folderRUNS + 'hist2d_MS_signal.pdf')

plot_scatter_hist2d(data['1']['DNNPredTrueClass'][(data['1']['DNNTrueClass'] == 0) & (data['1']['CCIsSS'] == 0)],
                    data['2']['DNNPredTrueClass'][(data['2']['DNNTrueClass'] == 0) & (data['1']['CCIsSS'] == 0)],
                    'U', 'V', 'gamma events', folderRUNS + 'hist2d_MS_background.pdf')

exit()

# plt.clf()
# plt.step(bin_centres, hist_y_SS, where='mid', color='firebrick', label='gamma')
# plt.step(bin_centres, hist_ee_SS, where='mid', color='blue', label='2beta')
# plt.xlabel('threshold')
# plt.legend(loc='best')
# plt.title('SS-only')
# plt.xlim([0, 1])
# plt.ylim(ymin=0)
# plt.savefig(args.folderOUT + 'histogram_SS_vs_threshold.pdf', bbox_inches='tight')
# plt.close()
#
# plt.clf()
# plt.step(bin_centres, hist_y_MS, where='mid', color='firebrick', label='gamma')
# plt.step(bin_centres, hist_ee_MS, where='mid', color='blue', label='2beta')
# plt.xlabel('threshold')
# plt.legend(loc='best')
# plt.title('MS-only')
# plt.xlim([0, 1])
# plt.ylim(ymin=0)
# plt.savefig(args.folderOUT + 'histogram_MS_vs_threshold.pdf', bbox_inches='tight')
# plt.close()

from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score

energies = np.linspace(1000, 3000, 5, endpoint=True)
eval_dict = {'cm': [], 'as': [], 'ps': [], 'rs': [], 'fs': [], 'prc': [], 'roc': [], 'roc_auc': []}
for i in range(len(energies)):
    if i == 0:
        mask = np.ones(data['QValue'].size, dtype=bool)
        print 'Validating energies: %.0f - %.0f' % (energies[0], energies[-1])
    else:
        mask = np.asarray((data['QValue'] >= energies[i - 1]) & (data['QValue'] < energies[i]))
        print 'Validating energies: %.0f - %.0f' % (energies[i - 1], energies[i]), '\tNumber of events:', (
        mask == True).sum()

    eval_dict['cm'].append(confusion_matrix(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
    print eval_dict['cm'][-1]
    tn, fp, fn, tp = eval_dict['cm'][-1].ravel()
    print 'true negative (y->y)\t', tn
    print 'true positive (ee->ee)\t', tp
    print 'false positive (y->ee)\t', fp
    print 'false negative (ee->y)\t', fn

    eval_dict['as'].append(accuracy_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
    eval_dict['ps'].append(precision_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
    eval_dict['rs'].append(recall_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
    eval_dict['fs'].append(f1_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))

    print 'accuracy score\t', eval_dict['as'][-1]
    print 'recall score\t', eval_dict['rs'][-1]
    print 'precision score\t', eval_dict['ps'][-1]
    print 'f1 score\t', eval_dict['fs'][-1]

    # cr = classification_report(data['DNNTrueClass'][mask], data['DNNPredClass'][mask], target_names=['gamma', 'bb'])
    # print cr

    eval_dict['prc'].append(
        precision_recall_curve(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))

    eval_dict['roc'].append(roc_curve(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))
    eval_dict['roc_auc'].append(
        roc_auc_score(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))
    print 'roc auc score\t', eval_dict['roc_auc'][-1]

    print '========================================================'

plt.clf()
for i in range(len(energies[:-1])):
    plt.plot(eval_dict['prc'][i + 1][2], eval_dict['prc'][i + 1][0][:-1], '--', color='C%d' % i)
    plt.plot(eval_dict['prc'][i + 1][2], eval_dict['prc'][i + 1][1][:-1], '-', color='C%d' % i,
             label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
plt.plot(eval_dict['prc'][0][2], eval_dict['prc'][0][0][:-1], 'k--', lw=2)
plt.plot(eval_dict['prc'][0][2], eval_dict['prc'][0][1][:-1], 'k-', lw=2, label='total')
plt.xlabel('threshold')
plt.legend(loc='lower left')
plt.title('- - - - precision   |   $^{\_\_\_\_}$ recall')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(args.folderOUT + 'precision_recall_vs_threshold.pdf', bbox_inches='tight')
plt.close()

plt.clf()
for i in range(len(energies[:-1])):
    plt.plot(eval_dict['prc'][i + 1][1], eval_dict['prc'][i + 1][0],
             label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
plt.plot(eval_dict['prc'][0][1], eval_dict['prc'][0][0], 'k-', lw=2, label='total')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc='best')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(args.folderOUT + 'precision_vs_recall.pdf', bbox_inches='tight')
plt.close()

plt.clf()
for i in range(len(energies[:-1])):
    plt.plot(eval_dict['roc'][i + 1][0], eval_dict['roc'][i + 1][1],
             label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
plt.plot(eval_dict['roc'][0][0], eval_dict['roc'][0][1], 'k-', lw=2, label='total')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig(args.folderOUT + 'roc_curve.pdf', bbox_inches='tight')
plt.close()

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------


#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        # ang = degrees(atan2(dy,dx))
        # ang = degrees(dy/dx)
        ang = 60.

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)