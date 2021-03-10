# load model to predict 2020 without train the entire neural networkimport matplotlib
import matplotlib
from keras.models import load_model
import winsound
import numpy as np
import matplotlib.pyplot as plt  # per creare i grafici
import matplotlib.dates as md
import time  # per vedere quanto impiega la rete neurale per giungere a soluzione
import pandas as pd
# summarize model.
# load dataset

def rmse(predictions, targets):
    differences = predictions - targets  # the DIFFERENCEs.

    differences_squared = differences ** 2  # the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)  # ROOT of ^

    return rmse_val  # get the ^
start = time.time()
cartella = "C:\\Users\\Gabri\\envs\\cer\\Scripts\\onde\\"
file = "30min_onde_corr_"
test2020 = pd.read_csv(cartella + file + "2020.txt", header=None, date_parser=[0])
PV_test = test2020[1].to_numpy()  # --Dati Ponteverdi per il test--#
Cas_test = test2020[2].to_numpy()  # --Dati Casalmaggiore per il test--#
Col_test = test2020[3].to_numpy()  # --Dati Colorno per il test--#
data_t = pd.to_datetime(test2020[0])  # --Date delle singole altezze idriche  per il test--#
onda1 = np.concatenate((np.where(data_t == "29-02-2020 00:00"), np.where(data_t == "18-03-2020 00:00")))
onda2 = np.concatenate((np.where(data_t == "03-12-2020 00:00"), np.where(data_t == "13-12-2020 00:00")))
test = 2
reqm = np.zeros(test)
maxerror = np.zeros(test)
# --Variabili interne ai cicli for--#
niter = 10 + 1  # int((((a - da) / passo) + 1)**2)  # --Quanti cicli for si effettueranno? meglio abbondare che deficere--#
rmse_test = np.zeros(niter)
acc = np.zeros(niter)
col = np.zeros(niter)
bat = np.zeros(niter)
ep = np.zeros(niter)
err = np.zeros(niter)
fin = np.zeros(niter)
cicl = np.zeros(niter)
neur = np.zeros(niter)
mederr = np.zeros(niter)
medrmse = np.zeros(test)
stdrmse = np.zeros(test)
reqm = np.zeros([test, niter - 1])
for k in range(1, test + 1, 1):
    exec(f'wave{k}= pd.DataFrame(np.zeros((onda{k}[1, 0] - onda{k}[0, 0], niter+1))).to_numpy()')
r = 0
REQM = np.zeros(test + 1)  # RMSE
ErMAX = np.zeros(test + 1)
STDM = np.zeros(test + 1)
temp = 18  # -- prendo in considerazione serie temporali di 18 ore--#
# -- Cicli for--#
print('fine caricamento dati')
shift = 6  # ore
shiftmez = shift * 2  # -- !!! Dipende dalla discretizzazione dei dati di input, ad ora di mezzora, nel caso bisognerà cambiare shiftmezz --#
column = (temp - shift) * 2  # -- Dati precedenti --#
# -- Preparo variabili --#
PV_testi = pd.DataFrame(np.zeros((len(PV_test) - column, column))).to_numpy()
Cas_testi = pd.DataFrame(np.zeros((len(Cas_test) - column, column))).to_numpy()
Col_testi = pd.DataFrame(np.zeros((len(Col_test) - column, column))).to_numpy()
for i in range(len(PV_test) - column):  # -- Preparo set dati input di test --#
    k = 0
    for j in range(column, 0, -1):
        PV_testi[i, k] = PV_test[j + i]
        Cas_testi[i, k] = Cas_test[j + i]
        Col_testi[i, k] = Col_test[j + i]
        k = k + 1
# -- Applico lo scostamento temporale per prevedere il futuro --#
y_test = pd.DataFrame(Col_test).iloc[(shiftmez + column):].to_numpy()
if shiftmez == 0:
    x_test = pd.DataFrame(np.hstack((PV_testi, Cas_testi, Col_testi))).to_numpy()
else:
    x_test = pd.DataFrame(np.hstack((PV_testi, Cas_testi, Col_testi))).iloc[:-shiftmez].to_numpy()
j = 0
# -- Scelta set iperparametri --#
for ciclo in range(1, niter, 1):
    model = load_model('b60n8shift' + str(shift) + '_' + str(ciclo) +'_model.h5')
    predictions = model.predict(x_test)
    o = np.zeros((len(y_test)))
    y_ = np.zeros((len(y_test)))
    for j in range(len(y_test)):
        o[j] = predictions[j]
        y_[j] = y_test[j]
    y_test = y_
    j = 0
    # -- Plot e calcoli sulle onde scelte--#
    # -- la  separazione è  dovuta alle differenti variabili per il set treaning e per il set test--#
    onda = {}
    for k in range(1, test + 1, 1):
        exec(f'onda = onda{k} ')
        j = 0
        ordinate1 = np.zeros(onda[1, 0] - onda[0, 0])
        ordinate2 = np.zeros(onda[1, 0] - onda[0, 0])
        datee = np.zeros(onda[1, 0] - onda[0, 0])
        ascisse = data_t.iloc[onda[0, 0]: onda[1, 0]]
        for ond in range(onda[0, 0], onda[1, 0], 1):
            ordinate1[j] = y_test[ond]
            ordinate2[j] = predictions[ond]
            datee[j] = j
            j = j + 1
        j = 0
        reqm[(k - 1), r] = rmse(ordinate2, ordinate1)
        exec(f'wave{k}[:, 0] = ordinate1')
        exec(f'wave{k}[:, ciclo] = ordinate2')
    # -- salvataggio variabili per txt riassuntivo--#
    rmse_test[r] = rmse(o, y_test)  # --Tutto il 18-19--#
    fin[r] = shift
    cicl[r] = ciclo
    r = r + 1
    print('GIRO: %.2f' % r)  # -- stampo a schermo la progressione dei i cicli for stanno progredendo --#
wave = {}
for t in range(1, test + 1, 1):
    exec(f'wave = wave{t} ')
    exec(f'onda = onda{t} ')
    j = 0
    ordinate1 = wave[:, 0]
    datee = np.zeros(onda[1, 0] - onda[0, 0])
    ordinate2 = np.zeros(len(wave))
    STD = np.zeros(len(wave))
    ascisse = data_t.iloc[onda[0, 0]: onda[1, 0]]
    dati = "new data"
    dates = matplotlib.dates.date2num(ascisse)
    ascisse = ascisse.array
    for q in range(0, len(wave), 1):
        datee[q] = q
        ru = pd.Series(wave[q, range(1, niter, 1)])
        ordinate2[q] = ru.mean()
        STD[q] = ru.std()
        exec(f'wave{t}[{q}, niter]= ordinate2[q]')
    ordinate1 = wave[:, 0]
    maxonda1 = max(ordinate1)
    maxonda2 = max(ordinate2)
    val = reqm[(t - 1), :]
    medrmse[t - 1] = val.mean()
    stdrmse[t - 1] = 2*val.std()
    REQM[t - 1] = rmse(ordinate1, ordinate2)  # (st.mean((ordinate1 - ordinate2) ** 2) ** 0.5)
    STDM[t - 1] = STD.mean()
    ErMAX[t - 1] = abs(maxonda1 - maxonda2)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    # xfmt = md.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(xfmt)
    line1 = plt.plot(datee, ordinate1, label='dati reali', ls='-', linewidth=1)
    line2 = plt.plot(datee, ordinate2, label='dati predetti', ls='-', linewidth=1)
    line3 = plt.plot(datee, (ordinate2 + STD+STD), label='2STD sup', ls=':', mec='lime', linewidth=0.7)
    line4 = plt.plot(datee, (ordinate2 - STD-STD), label='2STD inf', ls=':', mec='lime', linewidth=0.7)
    plt.annotate(dati + "  " + str(ascisse[0]) + '/ ' + str(ascisse[len(ascisse) - 1]), xy=(0.1, 0),
                 xycoords='axes fraction',
                 fontsize=9,
                 xytext=(200, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.annotate('onda n: ' + str(t), xy=(1, 0), xycoords='axes fraction',
                 fontsize=9,
                 xytext=(0, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.legend(loc="upper right")
    plt.grid()
    plt.ylabel('[m] da zero idrometrico')
    ax = plt.gca()
    table_data = [
        ["Previsione(h):" + str(shift), " Media [m]", "2 std [m]",
         "rmse [m]",
         "errore picchi [m]"],
        ["Singoli RMSE", np.around(medrmse[t - 1], 3), np.around(stdrmse[t - 1], 3), "/", "/"],
        ["Media valori", "/", np.around(STDM[t - 1], 3), np.around(REQM[t - 1], 3), np.around(ErMAX[t - 1], 3)]
    ]
    table = ax.table(cellText=table_data, loc='bottom', bbox=[0.0, -0.45, 1, .28])
    plt.subplots_adjust(bottom=0.2)
    table.set_fontsize(9)
    table.scale(1, 1)
    plt.savefig("Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + '_Batch' + str(
        60) + '_Neu' + str(8) + '.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    # -- Save onda in txt mode --#
    ondaa = "Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + "_Batch" + str(
        60) + "_Neu" + str(8) + ".txt"
    with open(ondaa, "w") as out_file:
        for i in range(len(ascisse)):
            out_string = ""
            out_string += str(ascisse[i])
            out_string += "," + str(ordinate1[i])  # --ordinate2 è il dato reale
            out_string += "," + str(ordinate2[i])  # --ordinate2 è la predizione
            out_string += "," + str(ordinate2[i] + 2 * STD[i])
            out_string += "," + str(ordinate2[i] - 2 * STD[i])
            out_string += "\n"
            out_file.write(out_string)

print('END, SAVE ALL RMSE')  # -- fine dei cicli for--#
end = time.time()
t = str((end - start) / 60)
print('tempo:' + t + ' [min] _end prediction model')  # -- stampa a schermo il tempo impiegato--#
dat = "Colorno3_Finestra(h)" + str(temp) + "_recap_Batch" + str(60) + "_Neu" + str(
    8) + ".txt"  # -- salvataggio in txt delle variabili riassuntivi dei cicli for--#
with open(dat, "w") as out_file:
    for t in range(1, test + 1, 1):
        out_string = ""
        out_string += str(t)
        out_string += "," + str(np.around(STDM[t - 1], 3))  # --std sulla media
        out_string += "," + str(np.around(REQM[t - 1], 3))  # --RMSE sulla media
        out_string += "," + str(np.around(medrmse[t - 1], 3))  # --media degli RMSE
        out_string += "," + str(np.around(stdrmse[t - 1], 3))  # --std sugli RMSE
        out_string += "\n"
        out_file.write(out_string)
frequency = 900  # Set Frequency To 2500 Hertz
duration = 900  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
winsound.Beep(frequency, duration)
end = time.time()
t = str((end - start) )
print('tempo:' + t + ' [sec] _end prediction model')  # -- stampa a schermo il tempo impiegato--#