import numpy as np
import matplotlib.pyplot as plt


def plot_traininghistory(folderOUT):

    data = np.loadtxt(folderOUT + 'training_history.csv', delimiter='\t', skiprows=1)

    try:

        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight')

        plt.clf()
        plt.plot(data[:, 0], data[:, 2])
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(folderOUT + 'plot_meanabserror.png', bbox_inches='tight')
        plt.clf()
        # plt.show()
        # plt.draw()
    except:
        plt.plot(data[0], data[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight')

        plt.clf()
        plt.plot(data[0], data[2])
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(folderOUT + 'plot_meanabserror.png', bbox_inches='tight')
        plt.clf()


# folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180613-1454-50/' #180606-1511-44/180607-1014-05'
# # folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180612-1614-49/'
#
# data1 = np.genfromtxt(folderIN + 'log_train.txt', delimiter='\t')   # , skiprows=1)
# data2 = np.loadtxt(folderIN + '180629-1422-34/' + 'log_train.txt', delimiter='\t', skiprows=1)
# data3 = np.loadtxt(folderIN + '180629-1422-34/180703-1103-15/' + 'log_train.txt', delimiter='\t', skiprows=1)
# # data4 = np.loadtxt(folderIN + '180613-1206-29/180613-1639-47/180614-1026-23/' + 'log_train.txt', delimiter='\t', skiprows=1)
#
#
# data = np.append(data1, data2, axis=0)
# data = np.append(data, data3, axis=0)
# # data = np.append(data, data4, axis=0)
# #
# # data = data1
#
# x = np.arange(len(data)) / 13000. * 500
# # x =
#
# fig, ax = plt.subplots()
# ax.semilogy(x, data[:, 2], label='Loss', color='blue')
# ax.semilogy(x, data[:, 4], label='Validation Loss', alpha=0.5, color='green')
# legend = ax.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig(folderIN + 'lossAndVallos.png', bbox_inches='tight')


# plt.clf()
# plt.semilogy(data[:, 0], data[:, 2])
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.savefig(folderOUT + 'total_mae.png', bbox_inches='tight')
#
# plt.clf()
# plt.semilogy(data[:, 0], data[:, 3])
# plt.xlabel('Epoch')
# plt.ylabel('Validation Loss')
# plt.savefig(folderOUT + 'total_val_loss.png', bbox_inches='tight')

