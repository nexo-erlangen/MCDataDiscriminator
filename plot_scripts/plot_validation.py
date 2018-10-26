#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
from plot_scripts.plot_traininghistory import *
BIGGER_SIZE = 13
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def validation_mc_plots(args, folderOUT, data):

    print data.keys()

    maskBDT = True
    for bdt_var in filter(lambda x: 'BDT' in x, data.keys()):
        maskBDT = maskBDT & (data[bdt_var] != -2.0)

    maskSS = maskBDT & (data['CCIsSS'] == 1)
    maskMS = np.invert(maskSS)

    maskROI = (np.sum(data['CCPurityCorrectedEnergy'], axis=1) > 2400.) & \
              (np.sum(data['CCPurityCorrectedEnergy'], axis=1) < 2800.)
    # maskROI = (np.sum(data['CCCorrectedEnergy'], axis=1) > 2400.) & \
    #           (np.sum(data['CCCorrectedEnergy'], axis=1) < 2800.)
    # maskROI = True

    maskBKG = (data['DNNTrueClass'] == 0)
    maskSIG = (data['DNNTrueClass'] == 1)

    # plot_energy_spectrum(fOUT=args.folderOUT + 'energy_spectrum_ROI.pdf',
    #                      data=[(np.sum(data['CCPurityCorrectedEnergy'], axis=1)[maskROI & maskSS & maskBKG]),
    #                            (np.sum(data['CCPurityCorrectedEnergy'], axis=1)[maskROI & maskSS & maskSIG])],
    #                      label=['Background', 'Signal'])
    #
    # plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve_BDT_SS.pdf',
    #                dataTrue=data['DNNTrueClass'][maskSS],
    #                dataPred=[data['DNNPredTrueClass'][maskSS],
    #                          data['BDT-SS-NoStandoff'][maskSS],
    #                          data['BDT-SS'][maskSS]],
    #                label=['DNN', 'BDT-Uni', 'BDT-Std'])
    # plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve_BDT_MS.pdf',
    #                dataTrue=data['DNNTrueClass'][(data['CCIsSS'] == 0)],
    #                dataPred=[data['DNNPredTrueClass'][(data['CCIsSS'] == 0)],
    #                          data['BDT-SSMS'][(data['CCIsSS'] == 0)]],
    #                label=['DNN', 'BDT-SSMS'])
    # plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve_BDT_SSMS.pdf',
    #                dataTrue=data['DNNTrueClass'],
    #                dataPred=[data['DNNPredTrueClass'],
    #                          data['BDT-SSMS']],
    #                label=['DNN', 'BDT-SSMS'])
    # exit()

    plot_energy_spectrum(fOUT=args.folderOUT + 'energy_spectrum_ROI.pdf',
                         data=[(np.sum(data['CCPurityCorrectedEnergy'], axis=1)[maskROI & maskSS & maskBKG]),
                               (np.sum(data['CCPurityCorrectedEnergy'], axis=1)[maskROI & maskSS & maskSIG])],
                         label=['Background', 'Signal'])

    plot_energy_spectrum(fOUT=args.folderOUT + 'energy_spectrum_MC_ROI.pdf',
                         data=[data['MCEnergy'][maskROI & maskSS & maskBKG],
                               data['MCEnergy'][maskROI & maskSS & maskSIG]],
                         label=['U238+Th232', 'bb0n'])

    plot_energy_spectrum(fOUT=args.folderOUT + 'standoff_SS_ROI.pdf',
                         data=[data['CCStandoff'][maskROI & maskSS & maskBKG],
                               data['CCStandoff'][maskROI & maskSS & maskSIG]],
                         label=['U238+Th232', 'bb0n'], xlabel='standoff distance [mm]')

    plot_energy_spectrum(fOUT=args.folderOUT + 'standoff_MS_ROI.pdf',
                         data=[data['CCStandoff'][maskROI & maskMS & maskBKG],
                               data['CCStandoff'][maskROI & maskMS & maskSIG]],
                         label=['U238+Th232', 'bb0n'], xlabel='standoff distance [mm]')

    plot_energy_spectrum(fOUT=args.folderOUT + 'discriminator_SS_ROI.pdf',
                         data=[data['BDT-DNN'][maskROI & maskSS & maskBKG],
                               data['BDT-DNN'][maskROI & maskSS & maskSIG]],
                         label=['Background', 'Signal'], xlabel='signal-likeness')

    plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve_ROI.pdf',
                   dataTrue=data['DNNTrueClass'][maskROI & maskSS],
                   dataPred=[data['DNNPredTrueClass'][maskROI & maskSS],
                             data['BDT-SS-Uni'][maskROI & maskSS],
                             data['BDT-SS-Std'][maskROI & maskSS],
                             -1.0 * np.sqrt(data['MCEventSizeR']**2+data['MCEventSizeZ']**2)[maskROI & maskSS]],
                   label=['DNN', 'BDT-Uni', 'BDT-Std', 'MC 3D Size'])

    plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve_ROI_BDT+DNN.pdf',
                   dataTrue=data['DNNTrueClass'][maskROI & maskSS],
                   dataPred=[data['DNNPredTrueClass'][maskROI & maskSS],
                             data['BDT-SS-Std'][maskROI & maskSS],
                             data['BDT-DNN'][maskROI & maskSS]],
                   label=['DNN', 'BDT-Std', 'DNN+Stand'])
    #
    # plot_prec_vs_recall_curve(fOUT=args.folderOUT + 'precision_vs_recall_ROI.pdf',
    #                           dataTrue=data['DNNTrueClass'][maskROI & maskSS],
    #                           dataPred=[data['DNNPredTrueClass'][maskROI & maskSS],
    #                                     norm_discriminator(data['BDT-SS-Uni'][maskROI & maskSS]),
    #                                     norm_discriminator(data['BDT-SS-Std'][maskROI & maskSS])],
    #                           label=['DNN', 'BDT-Uni', 'BDT-Std'])
    #
    # plot_prec_vs_recall_curve(fOUT=args.folderOUT + 'precision_vs_recall_MS_ROI.pdf',
    #                           dataTrue=data['DNNTrueClass'][maskROI & maskMS],
    #                           dataPred=[data['DNNPredTrueClass'][maskROI & maskMS],
    #                                     -1.0 * np.sqrt(data['MCEventSizeR'] ** 2 + data['MCEventSizeZ'] ** 2)[maskROI & maskMS]],
    #                           label=['DNN', 'MC Size (R)'])
    #
    # plot_prec_recall_vs_thresh_curve(fOUT=args.folderOUT + 'precision_recall_vs_threshold_ROI.pdf',
    #                                  dataTrue=data['DNNTrueClass'][maskROI & maskSS],
    #                                  dataPred=[data['DNNPredTrueClass'][maskROI & maskSS],
    #                                            norm_discriminator(data['BDT-SS-Uni'][maskROI & maskSS]),
    #                                            norm_discriminator(data['BDT-SS-Std'][maskROI & maskSS])],
    #                                  label=['DNN', 'BDT-Uni', 'BDT-Std'])

    # TODO BELOW IS TEST

    # threshold values for 90% signal efficiency
    bdt_ss_std = get_thresh_at_sig_eff_or_at_bkg_rej(data['DNNTrueClass'][maskROI & maskSS],
                                                     data['BDT-SS-Std'][maskROI & maskSS],
                                                     cut_value=0.9, mode='sig_eff')
    bdt_ss_uni = get_thresh_at_sig_eff_or_at_bkg_rej(data['DNNTrueClass'][maskROI & maskSS],
                                                     data['BDT-SS-Uni'][maskROI & maskSS],
                                                     cut_value=0.9, mode='sig_eff')
    dnn_ss_uni = get_thresh_at_sig_eff_or_at_bkg_rej(data['DNNTrueClass'][maskROI & maskSS],
                                                     data['DNNPredTrueClass'][maskROI & maskSS],
                                                     cut_value=0.9, mode='sig_eff')
    dnn_ss_stand = get_thresh_at_sig_eff_or_at_bkg_rej(data['DNNTrueClass'][maskROI & maskSS],
                                                     data['BDT-DNN'][maskROI & maskSS],
                                                     cut_value=0.9, mode='sig_eff')


    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventSize_BDT_Std_3D-90Sig.pdf',
                             data=np.sqrt(data['MCEventSizeR'] ** 2 + data['MCEventSizeZ'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Std'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_std,
                             discr_range=[0, 20], discr_label='BDT', data_label='True Event Size [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventSize_BDT_Uni_3D-90Sig.pdf',
                             data=np.sqrt(data['MCEventSizeR'] ** 2 + data['MCEventSizeZ'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Uni'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_uni,
                             discr_range=[0, 20], discr_label='BDT', data_label='True Event Size [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventSize_DNN_3D-90Sig.pdf',
                             data=np.sqrt(data['MCEventSizeR']**2+data['MCEventSizeZ']**2)[maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[0, 20], discr_label='DNN', data_label='True Event Size [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventSize_DNN_Stand_3D-90Sig.pdf',
                             data=np.sqrt(data['MCEventSizeR'] ** 2 + data['MCEventSizeZ'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-DNN'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_stand,
                             discr_range=[0, 20], discr_label='DNN+Stand', data_label='True Event Size [mm]')


    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_BDT_Std_R-90Sig.pdf',
                             data=np.sqrt(data['MCPosX'] ** 2 + data['MCPosY'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Std'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_std,
                             discr_range=[0, 180], discr_label='BDT', data_label='True Event Position R [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_BDT_Uni_R-90Sig.pdf',
                             data=np.sqrt(data['MCPosX'] ** 2 + data['MCPosY'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Uni'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_uni,
                             discr_range=[0, 180], discr_label='BDT', data_label='True Event Position R [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_R-90Sig.pdf',
                             data=np.sqrt(data['MCPosX'] ** 2 + data['MCPosY'] ** 2)[maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[0, 180], discr_label='DNN', data_label='True Event Position R [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_Stand_R-90Sig.pdf',
                             data=np.sqrt(data['MCPosX'] ** 2 + data['MCPosY'] ** 2)[maskROI & maskSS],
                             discr_mask=data['BDT-DNN'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_stand,
                             discr_range=[0, 180], discr_label='DNN+Stand', data_label='True Event Position R [mm]')


    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_BDT_Std_Z-90Sig.pdf',
                             data=data['MCPosZ'][maskROI & maskSS],
                             discr_mask=data['BDT-SS-Std'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_std,
                             discr_range=[-180, 180], discr_label='BDT', data_label='True Event Position Z [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_BDT_Uni_Z-90Sig.pdf',
                             data=data['MCPosZ'][maskROI & maskSS],
                             discr_mask=data['BDT-SS-Uni'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_uni,
                             discr_range=[-180, 180], discr_label='BDT', data_label='True Event Position Z [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_Z-90Sig.pdf',
                             data=data['MCPosZ'][maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[-180, 180], discr_label='DNN', data_label='True Event Position Z [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_Stand_Z-90Sig.pdf',
                             data=data['MCPosZ'][maskROI & maskSS],
                             discr_mask=data['BDT-DNN'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_stand,
                             discr_range=[-180, 180], discr_label='DNN+Stand', data_label='True Event Position Z [mm]')


    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_U-90Sig.pdf',
                             data=data['MCPosU'][maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[-180, 180], discr_label='DNN', data_label='True Event Position U [mm]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventPosition_DNN_V-90Sig.pdf',
                             data=data['MCPosV'][maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[-180, 180], discr_label='DNN', data_label='True Event Position V [mm]')


    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventEnergy_BDT_Std-90Sig.pdf',
                             data=(np.sum(data['CCCorrectedEnergy'], axis=1))[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Std'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_std,
                             discr_range=[2400, 2800], discr_label='BDT', data_label='uncalibrated corrected energy [keV]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventEnergy_BDT_Uni-90Sig.pdf',
                             data=(np.sum(data['CCCorrectedEnergy'], axis=1))[maskROI & maskSS],
                             discr_mask=data['BDT-SS-Uni'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=bdt_ss_uni,
                             discr_range=[2400, 2800], discr_label='BDT', data_label='uncalibrated corrected energy [keV]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventEnergy_DNN-90Sig.pdf',
                             data=(np.sum(data['CCCorrectedEnergy'], axis=1))[maskROI & maskSS],
                             discr_mask=data['DNNPredTrueClass'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_uni,
                             discr_range=[2400, 2800], discr_label='DNN', data_label='uncalibrated corrected energy [keV]')

    plot_Mikes_plot_idea_new(fOUT=args.folderOUT + 'EventEnergy_DNN_Stand-90Sig.pdf',
                             data=(np.sum(data['CCCorrectedEnergy'], axis=1))[maskROI & maskSS],
                             discr_mask=data['BDT-DNN'][maskROI & maskSS],
                             sig_or_bkg=maskSIG[maskROI & maskSS], discr_thresh=dnn_ss_stand,
                             discr_range=[2400, 2800], discr_label='DNN+Stand',
                             data_label='uncalibrated corrected energy [keV]')



    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_vs_threshold.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG],
                                      data['DNNPredTrueClass'][maskSIG]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_SS_vs_threshold.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskSS],
                                      data['DNNPredTrueClass'][maskSIG & maskSS]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_MS_vs_threshold.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskMS],
                                      data['DNNPredTrueClass'][maskSIG & maskMS]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_vs_threshold-ROI.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI],
                                      data['DNNPredTrueClass'][maskSIG & maskROI]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_SS_vs_threshold-ROI.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI & maskSS],
                                      data['DNNPredTrueClass'][maskSIG & maskROI & maskSS]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_MS_vs_threshold-ROI.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI & maskMS],
                                      data['DNNPredTrueClass'][maskSIG & maskROI & maskMS]],
                                label=['Background', 'Signal'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_vs_threshold-ROI-DNN-BDT.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI],
                                      data['DNNPredTrueClass'][maskSIG & maskROI],
                                      norm_discriminator(data['BDT-DNN'][maskBKG & maskROI]),
                                      norm_discriminator(data['BDT-DNN'][maskSIG & maskROI])],
                                label=['Background', 'Signal', 'Bkg (DNN+Stand)', 'Sig (DNN+Stand)'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_SS_vs_threshold-ROI-DNN-BDT.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI & maskSS],
                                      data['DNNPredTrueClass'][maskSIG & maskROI & maskSS],
                                      norm_discriminator(data['BDT-DNN'][maskBKG & maskROI & maskSS]),
                                      norm_discriminator(data['BDT-DNN'][maskSIG & maskROI & maskSS])],
                                label=['Background', 'Signal', 'Bkg (DNN+Stand)', 'Sig (DNN+Stand)'])

    plot_histogram_vs_threshold(fOUT=args.folderOUT + 'histogram_MS_vs_threshold-ROI-DNN-BDT.pdf',
                                data=[data['DNNPredTrueClass'][maskBKG & maskROI & maskMS],
                                      data['DNNPredTrueClass'][maskSIG & maskROI & maskMS],
                                      norm_discriminator(data['BDT-DNN'][maskBKG & maskROI & maskMS]),
                                      norm_discriminator(data['BDT-DNN'][maskSIG & maskROI & maskMS])],
                                label=['Background', 'Signal', 'Bkg (DNN+Stand)', 'Sig (DNN+Stand)'])

    plot_ROC_curve(fOUT=args.folderOUT + 'roc_curve.pdf',
                   dataTrue=[data['DNNTrueClass'][maskSS],
                             data['DNNTrueClass'][maskMS],
                             data['DNNTrueClass']],
                   dataPred=[data['DNNPredTrueClass'][maskSS],
                             data['DNNPredTrueClass'][maskMS],
                             data['DNNPredTrueClass']],
                   label=['DNN SS', 'DNN MS', 'DNN SS+MS'])


    exit()

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
        f1_score, accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score

    # energies = np.linspace(1000, 3000, 5, endpoint=True)
    energies = np.linspace(2400, 2700, 4, endpoint=True)
    eval_dict = {'cm': [], 'as': [], 'ps': [], 'rs': [], 'fs': [], 'prc': [], 'roc': [], 'roc_auc': []}
    for i in range(len(energies)):
        if i == 0:
            # mask = np.ones((np.sum(data['CCPurityCorrectedEnergy'], axis=1).size), dtype=bool)
            mask = data['CCIsSS'] == 1
            print 'Validating energies: %.0f - %.0f' % (energies[0], energies[-1])
        else:
            # continue
            mask = np.asarray((data['CCIsSS'] == 1) & (np.sum(data['CCPurityCorrectedEnergy'], axis=1) >= energies[i - 1]) & (np.sum(data['CCPurityCorrectedEnergy'], axis=1) < energies[i]))
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
    plt.plot([0, 1], [0.9, 0.9], 'k-', lw=0.5)
    for i in range(len(energies[:-1])):
        plt.plot(1.0-eval_dict['roc'][i + 1][0], eval_dict['roc'][i + 1][1],
                 label='%.1f-%.1f MeV (%.0f %%)' % (energies[i] / 1.e3, energies[i + 1] / 1.e3, 100.*(eval_dict['roc_auc'][i + 1]-0.5)))
    plt.plot(1.0-eval_dict['roc'][0][0], eval_dict['roc'][0][1], 'k-', lw=2, label='total (%.0f %%)' % (100.*(eval_dict['roc_auc'][0]-0.5)))
    plt.plot([1, 0], [0, 1], 'k--')
    plt.xlabel('Background rejection')
    plt.ylabel('Signal efficiency')
    plt.legend(loc='lower left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(args.folderOUT + 'roc_curve.pdf', bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def norm_discriminator(data, init=[-1,1], goal=[0,1]): #TODO Find equation to norm arbitrary ranges
    if not isinstance(data, np.ndarray):
        raise TypeError('wrong data type: %s'%(type(data)))
    return (data+1.0)/2.0

def get_thresh_at_sig_eff_or_at_bkg_rej(dataTrue, dataPred, cut_value=0.9, mode='sig_eff'):
    from sklearn.metrics import roc_curve
    if cut_value<0.0 or cut_value>1.0:
        raise ValueError('cut_value must be in range [0,1]. Used: %f'%(cut_value))
    fpr, tpr, thresh = roc_curve(dataTrue, dataPred)
    if mode=='sig_eff':
        idx = np.argmax(tpr > cut_value)
    elif mode=='bkg_rej':
        idx = np.argmax(fpr > cut_value)
    else:
        raise ValueError('mode must be sig_eff or bkg_rej. Used: %s' % (mode))
    print idx, thresh[idx], tpr[idx], fpr[idx]
    return thresh[idx]

# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, range_x, range_y, name_x, name_y, name_title, fOUT):
    # hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[range_x,range_y], bins=50, normed=True )
    # extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    gridsize = 100
    # plt.plot([0,1], [0,1], 'k--')
    plt.hexbin(E_x, E_y, extent=extent, gridsize=gridsize, mincnt=1, linewidths=0.1, cmap=plt.get_cmap('viridis'))  # norm=mpl.colors.LogNorm())

#   (gs, int(gs / (range_x[1] - range_x[0]) / (range_y[1] - range_y[0])))
    plt.title('%s' % (name_title))
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.xlim(xmin=range_x[0], xmax=range_x[1])
    plt.ylim(ymin=range_y[0], ymax=range_y[1])
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_ROC_curve(fOUT, dataTrue, dataPred, label):
    from sklearn.metrics import roc_curve, roc_auc_score

    if isinstance(dataPred, list): pass
    elif isinstance(dataPred, np.ndarray):
        dataPred = [dataPred]
        label = [label]
    else:
        raise TypeError('passed variable need to be list/np.ndarray')

    if isinstance(dataTrue, list): pass
    elif isinstance(dataTrue, np.ndarray):
        dataTrue = [dataTrue]*len(dataPred)
    else:
        raise TypeError('passed variable need to be list/np.ndarray')

    plt.clf()
    # plt.plot([0.9, 0.9], [0, 1], 'k-', lw=0.5)
    plt.plot([0, 1], [0.9, 0.9], 'k-', lw=0.5)
    for i in xrange(len(dataPred)):
        roc_i = roc_curve(dataTrue[i], dataPred[i])
        roc_auc_i = roc_auc_score(dataTrue[i], dataPred[i])
        plt.plot(1.0-roc_i[0], roc_i[1], label='%s (%.1f %%)' % (label[i], (roc_auc_i-0.5) * 100.))
    plt.plot([1, 0], [0, 1], 'k--')
    plt.xlabel('Background rejection')
    plt.ylabel('Signal efficiency')
    plt.legend(loc='lower left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_histogram_vs_threshold(fOUT, data, label):
    if isinstance(data, list): pass
    elif isinstance(data, np.ndarray):
        data = [data]
        label = [label]
    else: raise TypeError('passed variable need to be list/np.ndarray')

    kwargs = {
        'range': (0, 1),
        'bins': 100,
        'density': True,
        'facecolor': 'None',
        'histtype': 'step',
        'lw': 2.
    }

    plt.clf()
    for i in xrange(len(data)):
        plt.hist(data[i], label='%s' % (label[i]), **kwargs)
    plt.xlabel('signal-likeness')
    plt.legend(loc='best')
    # plt.legend(loc='upper left')
    plt.xlim([0, 1])
    plt.ylim(ymin=0)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_prec_vs_recall_curve(fOUT, dataTrue, dataPred, label):
    from sklearn.metrics import precision_recall_curve

    if isinstance(dataPred, list): pass
    elif isinstance(dataPred, np.ndarray):
        dataPred = [dataPred]
        label = [label]
    else: raise TypeError('passed variable need to be list/np.ndarray')

    plt.clf()
    for i in xrange(len(dataPred)):
        pre_i, rec_i, thr_i = precision_recall_curve(dataTrue, dataPred[i])
        plt.plot(rec_i, pre_i, label='%s' % (label[i]))
    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_prec_recall_vs_thresh_curve(fOUT, dataTrue, dataPred, label):
    from sklearn.metrics import precision_recall_curve

    if isinstance(dataPred, list):
        pass
    elif isinstance(dataPred, np.ndarray):
        dataPred = [dataPred]
        label = [label]
    else:
        raise TypeError('passed variable need to be list/np.ndarray')

    plt.clf()
    for i in xrange(len(dataPred)):
        pre_vs_rec_i = precision_recall_curve(dataTrue, dataPred[i])
        plt.plot(pre_vs_rec_i[2], pre_vs_rec_i[0][:-1], '--', color='C%d' % i)
        plt.plot(pre_vs_rec_i[2], pre_vs_rec_i[1][:-1], '-' , color='C%d' % i, label='%s' % (label[i]))
    plt.xlabel('threshold')
    plt.title('- - - - precision   |   $^{\_\_\_\_}$ sensitivity')
    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_Mikes_plot_idea(fOUT, data, discr_mask, discr_criterion, bkg_or_sig, discr_range=[0, 20], discr_label='True Event Size [mm]'):
    if bkg_or_sig in ['bkg', 'Bkg', 'background', 'Background']:
        label = 'Background'
    elif bkg_or_sig in ['sig', 'Sig', 'signal', 'Signal']:
        label = 'Signal'
    else: raise ValueError('bkg_or_sig variable holds strange argument: %s'%(str(bkg_or_sig)))
    plt.figure(20)
    plt.subplot(211)

    bins = np.linspace(discr_range[0], discr_range[-1], 40)

    hist_full = plt.hist(data,  bins=bins, facecolor="None", color='r', linewidth=2.5, histtype='step', label='All %s'%(label), normed=False)
    hist_cut = plt.hist(data[discr_mask], bins=bins, facecolor="b", color='b', alpha=0.5, linewidth=2.5, histtype='stepfilled', label='%s (%s)'%(label, discr_criterion), normed=False)
    plt.ylabel("Counts [#]", fontsize=15)
    plt.grid(True)
    plt.xlim(bins[0], bins[-1])
    # plt.legend(loc='upper right')
    plt.legend(loc='best')

    plt.subplot(212)
    data_points = np.asarray(hist_full[1])[:-1] + np.diff(hist_full[1])[0]/2.0
    data_y = (hist_cut[0]*1.0)/hist_full[0]
    data_y_error = data_y * np.sqrt(1./hist_cut[0] + 1./hist_full[0])
    plt.fill_between(data_points, data_y - data_y_error, data_y + data_y_error, edgecolor='none', facecolor='k', alpha=0.8)
    plt.xlabel('%s'%discr_label, fontsize=15)
    plt.ylabel("Frac Cut (%s)"%(discr_criterion), fontsize=15)
    plt.xlim(bins[0], bins[-1])
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_Mikes_plot_idea_new(fOUT, data, discr_mask, sig_or_bkg, discr_thresh, discr_range, discr_label, data_label):
    labelBkg = 'Background'
    labelSig = 'Signal'

    bins = np.linspace(discr_range[0], discr_range[-1], 40)

    f = plt.figure(figsize=(6.4, 4.8*1.5))
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    hist_sig_all = ax1.hist(data[sig_or_bkg],  bins=bins, facecolor="None", color='k', linewidth=2.5, histtype='step', label='All %s'%(labelSig), normed=False)
    hist_sig_cut = ax1.hist(data[sig_or_bkg & (discr_mask>discr_thresh)], bins=bins, facecolor="b", color='b', alpha=0.5, linewidth=2.5, histtype='stepfilled', label='%s (%s>%.2f)'%(labelSig, discr_label, discr_thresh), normed=False)
    ax1.set_ylabel("Counts [#]", fontsize=13)
    ax1.grid(True)
    ax1.set_xlim(bins[0], bins[-1])
    ax1.legend(loc='best')
    plt.setp(ax1.get_xticklabels(), visible=False)

    hist_bkg_all = ax2.hist(data[np.invert(sig_or_bkg)],  bins=bins, facecolor="None", color='k', linewidth=2.5, histtype='step', normed=False)
    hist_bkg_cut = ax2.hist(data[np.invert(sig_or_bkg) & (discr_mask<discr_thresh)], bins=bins, facecolor="g", color='g', alpha=0.5, linewidth=2.5, histtype='stepfilled', label='%s (%s<%.2f)'%(labelBkg, discr_label, discr_thresh), normed=False)
    ax2.set_ylabel("Counts [#]", fontsize=13)
    ax2.grid(True)
    ax2.legend(loc='best')
    plt.setp(ax2.get_xticklabels(), visible=False)

    data_points = np.asarray(hist_sig_all[1])[:-1] + np.diff(hist_sig_all[1])[0]/2.0
    frac_sig = (hist_sig_cut[0] * 1.0) / hist_sig_all[0]
    frac_bkg = (hist_bkg_cut[0] * 1.0) / hist_bkg_all[0]
    frac_sig_error = frac_sig * np.sqrt(1. / hist_sig_cut[0] + 1. / hist_sig_all[0])
    frac_bkg_error = frac_bkg * np.sqrt(1. / hist_bkg_cut[0] + 1. / hist_bkg_all[0])
    ax3.fill_between(data_points, frac_bkg - frac_bkg_error, frac_bkg + frac_bkg_error, label=labelBkg, edgecolor='none', facecolor='g', alpha=0.8)
    ax3.fill_between(data_points, frac_sig - frac_sig_error, frac_sig + frac_sig_error, label=labelSig, edgecolor='none', facecolor='b', alpha=0.8)
    ax3.set_xlabel('%s'%data_label, fontsize=13)
    ax3.set_ylabel("Fraction Cut/All", fontsize=13)
    ax3.set_ylim(0.0, 1.0)
    plt.grid(True)

    f.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def plot_energy_spectrum(fOUT, data, label, xlabel='uncalibrated energy [keV]'):
    if isinstance(data, list):
        pass
    elif isinstance(data, np.ndarray):
        data = [data]
        label = [label]
    else:
        raise TypeError('passed variable need to be list/np.ndarray')

    plt.clf()
    rmin = min([np.min(d) for d in data])
    rmax = max([np.max(d) for d in data])
    print rmin, rmax
    for i in xrange(len(data)):
        plt.hist(data[i], bins=100, range=(rmin, rmax), facecolor="None", histtype='step', lw=2., label='%s' % (label[i]), density=True)
    plt.xlabel(xlabel)
    plt.ylabel('normed counts [#]')
    plt.legend(loc='best')
    plt.xlim([rmin, rmax])
    # plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# scatter
def plot_scatter(E_x, E_y, name_x, name_y, fOUT):
    dE = E_x - E_y
    diag = np.arange(min(E_x),max(E_x))
    plt.scatter(E_x, E_y, label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)))
    plt.plot(diag, diag, 'k--')
    plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

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

# ----------------------------------------------------------
# Final Plots
# ----------------------------------------------------------
def plot_learning_curve(folderOUT, data):
    plt.clf()
    plt.plot(data[1], data[4], label='Validation')
    plt.plot(data[1], data[2], label='Training')
    plt.xlabel('Training time [epoch]')
    plt.ylabel('Loss')
    plt.grid(True, which='both')
    plt.xlim(xmin=0)
    # plt.gca().set_yscale('log')
    plt.legend(loc="best")
    plt.savefig(folderOUT + 'loss.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.axhline(y=50, color='k')
    plt.plot(data[1], 100. * data[5], label='Validation')
    plt.plot(data[1], 100.*data[3], label='Training')
    plt.grid(True)
    plt.ylim(ymin=40.0, ymax=100.0)
    plt.xlim(xmin=0)
    plt.legend(loc="best")
    plt.xlabel('Training time [epoch]')
    plt.ylabel('Accuracy [%]')
    plt.savefig(folderOUT + 'accuracy.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

# ----------------------------------------------------------
# Math Functions
# ----------------------------------------------------------
def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def erf(x, mu, sigma, B):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)