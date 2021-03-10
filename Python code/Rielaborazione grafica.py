##
# first neural network with keras tutorial
import matplotlib
import winsound
import numpy as np
import matplotlib.pyplot as plt  # per creare i grafici
import matplotlib.dates as md
import time  # per vedere quanto impiega la rete neurale per giungere a soluzione
import statistics as st
import pandas as pd
from matplotlib.font_manager import FontProperties


def rmse(predictions, targets):
    differences = predictions - targets  # the DIFFERENCEs.

    differences_squared = differences ** 2  # the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)  # ROOT of ^

    return rmse_val  # get the ^
sec = [2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #num =13
#sec = [100, 80, 60, 40, 20, 8, 4, 2]
#sec = [1, 2, 4, 8, 10, 20, 30]
num = 13
for t in range(1, 19, 1):  # onde
    ErMAX = np.zeros(num)
    REQM = np.zeros(num)
    STDM = np.zeros(num)
    medrmse = np.zeros(num)
    stdrmse = np.zeros(num)
    j = 1
   # cartella = "C:\\Users\\Gabri\\envs\\cer\\Scripts\\shift9_neu2-4-8-10_100\\"
    cartella = "C:\\Users\\Gabri\\envs\\cer\\Scripts\\"
    #onda = pd.read_csv(
     #   cartella + "Onda_Col" + str(t) + "_prediz(h)9_cicli10_Batch" + str(20) + "_Neu" + str(80) + ".txt",
      #  header=None, date_parser=[0])
    shift = 6
    temp = 18
    batch = 80
    neu = 8
    onda = pd.read_csv(cartella + "Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + "_Batch"+ str(batch) +"_Neu" + str(neu) + ".txt",
                       header=None, date_parser=[0])
   # recap = pd.read_csv(cartella + "Colorno3_cicli10" + "_recap_Batch" + str(20) + "_Neu" + str(80) +".txt", header=None,
    #                    date_parser=[0])
    recap = pd.read_csv(cartella + "Colorno3_Finestra(h)" + str(temp) + "_recap_Batch" + str(batch) + "_Neu" + str(neu) + ".txt", header=None,
                        date_parser=[0])
    ordinate = onda
    ascisse = onda[0]
    datee = np.zeros(len(onda[1]))
    ordinate[0] = (onda[1])
    for batch in sec:  # --Neuroni--#

        # onda = pd.read_csv(
        #   cartella + "Onda_Col" + str(t) + "_prediz(h)9_cicli10_Batch" + str(20) + "_Neu" + str(80) + ".txt",
        #  header=None, date_parser=[0])
        onda = pd.read_csv(cartella + "Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + "_Batch"+ str(batch) +"_Neu" + str(neu) + ".txt",
                           header=None, date_parser=[0])
        # recap = pd.read_csv(cartella + "Colorno3_cicli10" + "_recap_Batch" + str(20) + "_Neu" + str(80) +".txt", header=None,
        #                    date_parser=[0])
        recap = pd.read_csv(cartella + "Colorno3_Finestra(h)" + str(temp) + "_recap_Batch" + str(batch) + "_Neu" + str(neu) + ".txt",
                            header=None,
                            date_parser=[0])
        ordinate[j] = onda.iloc[:, 2]
        ErMAX[j - 1] = abs(max(ordinate[0]) - max(ordinate[j]))
        # re = recap.iloc[t-1, 2]
        REQM[j - 1] = recap.iloc[t - 1, 2]
        # ri = recap.iloc[t-1, 1]
        STDM[j - 1] = recap.iloc[t - 1, 1]
        medrmse[j - 1] = recap.iloc[t - 1, 3]
        stdrmse[j - 1] = recap.iloc[t - 1, 4]
        j = j + 1
    for q in range(0, len(onda[0]), 1):
        datee[q] = q
    ordinate.to_csv(r'Onda_' + str(t) + 'confronto_valori_medishift9' + '_10Cicli_neu_Finestra(h)' + str(temp)+'.txt', index=True, header=True)
    ascisse = pd.to_datetime(ascisse)
    ascisse = ascisse.array
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    variabile = 'batch'
    #line1 = plt.plot(datee, ordinate[1], label='1 ' + variabile, ls='-', linewidth=1)
    line2 = plt.plot(datee, ordinate[2], label='2 ' + variabile, ls='-', linewidth=1)
    line3 = plt.plot(datee, ordinate[3], label='4 ' + variabile, ls='-', linewidth=1)
    line4 = plt.plot(datee, ordinate[4], label='8 ' + variabile, ls='-', linewidth=1)
    line5 = plt.plot(datee, ordinate[5], label='10 ' + variabile, ls='-', linewidth=1)
    line6 = plt.plot(datee, ordinate[6], label='20 ' + variabile, ls='-', linewidth=1)
    line7 = plt.plot(datee, ordinate[7], label='30 ' + variabile, ls='-', linewidth=1)
    line8 = plt.plot(datee, ordinate[8], label='40 ' + variabile, ls='-', linewidth=1)
    line9 = plt.plot(datee, ordinate[8], label='50 ' + variabile, ls='-', linewidth=1)
    line10 = plt.plot(datee, ordinate[8], label='60 ' + variabile, ls='-', linewidth=1)
    line11 = plt.plot(datee, ordinate[8], label='70 ' + variabile, ls='-', linewidth=1)
    line12 = plt.plot(datee, ordinate[8], label='80 ' + variabile, ls='-', linewidth=1)
    line13 = plt.plot(datee, ordinate[8], label='90 ' + variabile, ls='-', linewidth=1)
    line14 = plt.plot(datee, ordinate[8], label='100 ' + variabile, ls='-', linewidth=1)
    line0 = plt.plot(datee, ordinate[0], label='dati reali', ls='-', color='k', linewidth=1)
    plt.annotate(str(ascisse[0]), xy=(0.1, 0), xycoords='axes fraction',
                 fontsize=11,
                 xytext=(100, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.annotate('/ ' + str(ascisse[len(ascisse) - 1]), xy=(0.1, 0),
                 xycoords='axes fraction',
                 fontsize=11,
                 xytext=(230, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.annotate('onda n: ' + str(t), xy=(1, 0), xycoords='axes fraction',
                 fontsize=11,
                 xytext=(0, -25), textcoords='offset points',
                 ha='right', va='top')
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.grid()
    plt.ylabel('m')
    ax = plt.gca()
    table_data = [
        [variabile, "rmse [m]", "errore picco [m]", "media std [m]", "rmse medio[m]",
         "rmse std [m]"],
        ['2', np.around(REQM[0], 4), np.around(ErMAX[0], 4), np.around(STDM[0], 4), np.around(medrmse[0], 4),
         np.around(stdrmse[0], 4)],
        ['4', np.around(REQM[1], 4), np.around(ErMAX[1], 4), np.around(STDM[1], 4), np.around(medrmse[1], 4),
         np.around(stdrmse[1], 4)],
        ['8', np.around(REQM[2], 4), np.around(ErMAX[2], 4), np.around(STDM[2], 4), np.around(medrmse[2], 4),
         np.around(stdrmse[2], 4)],
        ['10', np.around(REQM[3], 4), np.around(ErMAX[3], 4), np.around(STDM[3], 4), np.around(medrmse[3], 4),
         np.around(stdrmse[3], 4)],
        ['20', np.around(REQM[4], 4), np.around(ErMAX[4], 4), np.around(STDM[4], 4), np.around(medrmse[4], 4),
         np.around(stdrmse[4], 4)],
        ['30', np.around(REQM[5], 4), np.around(ErMAX[5], 4), np.around(STDM[5], 4), np.around(medrmse[5], 4),
         np.around(stdrmse[5], 4)],
        ['40', np.around(REQM[6], 4), np.around(ErMAX[6], 4), np.around(STDM[6], 4), np.around(medrmse[6], 4),
         np.around(stdrmse[6], 4)],
         ['50',  np.around(REQM[7], 4), np.around(ErMAX[7], 4), np.around(STDM[7], 4), np.around(medrmse[7], 4),
          np.around(stdrmse[7], 4)],
         ['60', np.around(REQM[8], 4), np.around(ErMAX[8], 4), np.around(STDM[8], 4), np.around(medrmse[8], 4),
          np.around(stdrmse[8], 4)],
         ['70', np.around(REQM[9], 4), np.around(ErMAX[9], 4), np.around(STDM[9], 4), np.around(medrmse[9], 4),
          np.around(stdrmse[9], 4)],
         ['80', np.around(REQM[10], 4), np.around(ErMAX[10], 4), np.around(STDM[10], 4), np.around(medrmse[10], 4),
          np.around(stdrmse[8], 4)],
         ['90', np.around(REQM[10], 4), np.around(ErMAX[10], 4), np.around(STDM[10], 4), np.around(medrmse[10], 4),
          np.around(stdrmse[10], 4)],
         ['100', np.around(REQM[11], 4), np.around(ErMAX[11], 4), np.around(STDM[11], 4), np.around(medrmse[11], 4),
          np.around(stdrmse[11], 4)],
         #['100', np.around(REQM[12], 4), np.around(ErMAX[12], 4), np.around(STDM[12], 4), np.around(medrmse[12], 4),
          #np.around(stdrmse[12], 4)]
    ]
    table = ax.table(cellText=table_data, loc='bottom', bbox=[0.0, -0.65, 1, .48])
    plt.subplots_adjust(bottom=0.2)
    table.set_fontsize(13)
    table.scale(1, 1)
    plt.savefig('Onda_' + str(t) + 'confronto_valori_medishift6' + '_'+ variabile +'_var_Finestra(h)' + str(temp)+'.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    gf = pd.DataFrame(
        {'rmse [m]': REQM,
         'errore picco [m]': ErMAX,
         "media std [m]": STDM,
         'rmse medio[m]': medrmse,
         'rmse std [m]': stdrmse},
        index=['2', '4', '8', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        #index=['1', '2', '4', '8', '10', '20', '30'])
    gf.to_csv(r'Tab' + str(t) + 'confronto_valori_medishift6' + '_10Cicli_'+ variabile +'_Finestra(h)' + str(temp)+'.txt', index=True, header=True)
