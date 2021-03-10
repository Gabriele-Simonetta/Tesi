##
# first neural network with keras tutorial
import matplotlib
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import winsound
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import var
import matplotlib.pyplot as plt  # per creare i grafici
import matplotlib.dates as md
import time  # per vedere quanto impiega la rete neurale per giungere a soluzione
import statistics as st
import pandas as pd

def rmse(predictions, targets):
    differences = predictions - targets  # the DIFFERENCEs.

    differences_squared = differences ** 2  # the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)  # ROOT of ^

    return rmse_val  # get the ^
start = time.time()
# load the dataset
cartella = "C:\\Users\\Gabri\\envs\\cer\\Scripts\\onde\\"
file = "30min_onde_mod_" #"30min_onde_mod14_17_"
data2012 = pd.read_csv(cartella + file + "2012.txt", header=None, date_parser=[0])
data2013 = pd.read_csv(cartella + file + "2013.txt", header=None, date_parser=[0])
data2014 = pd.read_csv(cartella + file + "2014.txt", header=None, date_parser=[0])
data2015 = pd.read_csv(cartella + file + "2015.txt", header=None, date_parser=[0])
data2016 = pd.read_csv(cartella + file + "2016.txt", header=None, date_parser=[0])
data2017 = pd.read_csv(cartella + file + "2017.txt", header=None, date_parser=[0])
test2018 = pd.read_csv(cartella + file + "2018.txt", header=None, date_parser=[0])
test2019 = pd.read_csv(cartella + file + "2019.txt", header=None, date_parser=[0])
PV = pd.concat((data2012[1], data2013[1], data2014[1], data2015[1], data2016[1],
                data2017[1])).to_numpy()  # --Dati Ponteverdi per il treaning--#
Cas = pd.concat((data2012[2], data2013[2], data2014[2], data2015[2], data2016[2],
                 data2017[2])).to_numpy()  # --Dati Casalmaggiore per il treaning--#
Col = pd.concat((data2012[3], data2013[3], data2014[3], data2015[3], data2016[3],
                 data2017[3])).to_numpy()  # --Dati Colorno per il treaning--#
data = pd.concat((pd.to_datetime(data2012[0]), pd.to_datetime(data2013[0]), pd.to_datetime(data2014[0]),
                  pd.to_datetime(data2015[0]), pd.to_datetime(data2016[0]),
                  pd.to_datetime(data2017[0])))  # --Date delle singole altezze--#
PV_test = pd.concat((test2018[1], test2019[1])).to_numpy()  # --Dati Ponteverdi per il test--#
Cas_test = pd.concat((test2018[2], test2019[2])).to_numpy()  # --Dati Casalmaggiore per il test--#
Col_test = pd.concat((test2018[3], test2019[3])).to_numpy()  # --Dati Colorno per il test--#
data_t = pd.concat(
    (pd.to_datetime(test2018[0]), pd.to_datetime(test2019[0])))  # --Date delle singole altezze idriche  per il test--#

# -- Validation data --#
tracce = 1
datavalidation0 = np.zeros((2, 1))
datavalidation0 = np.concatenate((np.where(data == "01-06-2012 00:00"), np.where(data == "01-02-2013 00:00")))
datavalidation = np.zeros(0)
data_tre = pd.Series.tolist(data)
j = 0
for o in range(0, tracce, 1):
    exec(f'datavalidation = datavalidation{o} ')
    del data_tre[datavalidation[0, 0]: datavalidation[1, 0]]
data_tre = pd.Series(data_tre)
# ---Selezione onde--#
onda1 = np.concatenate((np.where(data_tre == "29-04-2012 00:00"), np.where(data_tre == "17-05-2012 00:00")))
onda2 = np.concatenate((np.where(data_tre == "19-04-2013 00:00"), np.where(data_tre == "08-06-2013 00:00")))
onda3 = np.concatenate((np.where(data_tre == "04-11-2014 00:00"), np.where(data_tre == "26-12-2014 00:00")))
onda4 = np.concatenate((np.where(data_tre == "22-03-2015 00:00"), np.where(data_tre == "8-04-2015 00:00")))
onda5 = np.concatenate((np.where(data_tre == "08-01-2016 00:00"), np.where(data_tre == "17-01-2016 00:00")))
onda6 = np.concatenate((np.where(data_tre == "05-11-2016 00:00"), np.where(data_tre == "10-11-2016 00:00")))
onda7 = np.concatenate((np.where(data_t == "04-03-2018 00:00"), np.where(data_t == "27-03-2018 00:00")))
onda8 = np.concatenate((np.where(data_t == "03-04-2018 00:00"), np.where(data_t == "18-04-2018 00:00")))
onda9 = np.concatenate((np.where(data_t == "21-06-2018 00:00"), np.where(data_t == "01-12-2018 00:00")))
onda10 = np.concatenate((np.where(data_t == "29-01-2019 00:00"), np.where(data_t == "16-02-2019 00:00")))
onda11 = np.concatenate((np.where(data_t == "22-04-2019 00:00"), np.where(data_t == "03-05-2019 00:00")))
onda12 = np.concatenate((np.where(data_t == "03-05-2019 00:00"), np.where(data_t == "11-06-2019 00:00")))
onda13 = np.concatenate((np.where(data_t == "21-10-2019 00:00"), np.where(data_t == "01-11-2019 00:00")))
onda14 = np.concatenate((np.where(data_t == "03-11-2019 00:00"), np.where(data_t == "08-11-2019 00:00")))
onda15 = np.concatenate((np.where(data_t == "14-11-2019 00:00"), np.where(data_t == "13-12-2019 00:00")))
onda16 = np.concatenate((np.where(data_t == "18-12-2019 00:00"), np.where(data_t == "01-01-2020 00:00")))
onda17 = np.concatenate((np.where(data_t == "12-10-2014 00:30"), np.where(data_t == "22-10-2014 00:00")))
#onda18 = np.concatenate((np.where(data_t == "08-12-2017 00:00"), np.where(data_t == "22-12-2017 23:30")))

training = 6
test = 17
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
PV_tren = pd.DataFrame(np.zeros((len(PV) - column, column))).to_numpy()
Cas_tren = pd.DataFrame(np.zeros((len(Cas) - column, column))).to_numpy()
Col_tren = pd.DataFrame(np.zeros((len(Col) - column, column))).to_numpy()
PV_testi = pd.DataFrame(np.zeros((len(PV_test) - column, column))).to_numpy()
Cas_testi = pd.DataFrame(np.zeros((len(Cas_test) - column, column))).to_numpy()
Col_testi = pd.DataFrame(np.zeros((len(Col_test) - column, column))).to_numpy()
X_val = pd.DataFrame(np.zeros((int(
    (datavalidation0[1, 0] - datavalidation0[0, 0])), column * 3))).to_numpy()
Y_val = pd.DataFrame(np.zeros((int(
    (datavalidation0[1, 0] - datavalidation0[0, 0])), 1))).to_numpy()
for i in range(len(PV) - column):  # -- Preparo set dati input di training --#
    k = 0
    for j in range(column, 0, -1):
        PV_tren[i, k] = PV[j + i]
        Cas_tren[i, k] = Cas[j + i]
        Col_tren[i, k] = Col[j + i]
        k = k + 1
for i in range(len(PV_test) - column):  # -- Preparo set dati input di test --#
    k = 0
    for j in range(column, 0, -1):
        PV_testi[i, k] = PV_test[j + i]
        Cas_testi[i, k] = Cas_test[j + i]
        Col_testi[i, k] = Col_test[j + i]
        k = k + 1
# -- Applico lo scostamento temporale per prevedere il futuro --#
Y_tr = pd.DataFrame(Col).iloc[(shiftmez + column):]
y_test = pd.DataFrame(Col_test).iloc[(shiftmez + column):].to_numpy()
if shiftmez == 0:
    X_tr = pd.DataFrame(np.hstack((PV_tren, Cas_tren, Col_tren)))
    x_test = pd.DataFrame(np.hstack((PV_testi, Cas_testi, Col_testi))).to_numpy()
else:
    X_tr = pd.DataFrame(np.hstack((PV_tren, Cas_tren, Col_tren))).iloc[:-shiftmez]
    x_test = pd.DataFrame(np.hstack((PV_testi, Cas_testi, Col_testi))).iloc[:-shiftmez].to_numpy()
X_tra = X_tr.to_numpy()
Y_tra = Y_tr.to_numpy()
j = 0
for o in range(0, tracce, 1):
    exec(f'datavalidation = datavalidation{o} ')
    for ond in range(datavalidation[0, 0], datavalidation[1, 0], 1):
        X_val[j, :] = X_tra[ond, :]
        Y_val[j] = Y_tra[ond]
        X_tr = X_tr.drop(ond)
        Y_tr = Y_tr.drop(ond)
        j = j + 1
Y_tr = Y_tr.to_numpy()
X_tr = X_tr.to_numpy()
# -- Scelta set iperparametri --#
batch = 60  # b
neu = 8  # n
epoche = 100  # e
for ciclo in range(1, niter, 1):
    # -- define the keras model --#
    model = Sequential()
    model.add(Dense(neu, input_dim=(column * 3), activation='relu'))
    model.add(Dense(1, activation='linear'))
    # -- compile the keras model --#
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # -- fit the keras model on the dataset --#
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val), epochs=epoche, batch_size=batch,
              verbose=0, callbacks=[es])
    # -- evaluate the keras model --#
    _, accuracy = model.evaluate(X_tr, Y_tr)
    print('Accuracy: %.2f ' % (accuracy * 100), '%', '_end compile model')
    # --  make class predictions with the model --#
    predictions = model.predict(x_test)
    predizione = model.predict(X_tr)
    # -- Rimodellazione varibili per usarle nei grafici --#
    o = np.zeros((len(y_test)))
    y_ = np.zeros((len(y_test)))
    p = np.zeros((len(Y_tr)))
    Y_ = np.zeros((len(Y_tr)))
    for j in range(len(y_test)):
        o[j] = predictions[j]
        y_[j] = y_test[j]
    for j in range(len(Y_tr)):
        p[j] = predizione[j]
        Y_[j] = Y_tr[j]
    Y_tr = Y_
    y_test = y_
    j = 0
    # -- Plot e calcoli sulle onde scelte--#
    # -- la  separazione è  dovuta alle differenti variabili per il set treaning e per il set test--#
    onda = {}
    for k in range(1, training + 1, 1):
        exec(f'onda = onda{k} ')
        ordinate1 = np.zeros(onda[1, 0] - onda[0, 0])
        ordinate2 = np.zeros(onda[1, 0] - onda[0, 0])
        datee = np.zeros(onda[1, 0] - onda[0, 0])
        ascisse = data_tre.iloc[onda[0, 0]: onda[1, 0]]
        for ond in range(onda[0, 0], onda[1, 0], 1):
            ordinate1[j] = Y_tr[ond - shiftmez]
            ordinate2[j] = predizione[ond - shiftmez]
            datee[j] = j
            j = j + 1
        j = 0
        reqm[(k - 1), r] = rmse(ordinate2, ordinate1)
        exec(f'wave{k}[:, 0] = ordinate1')
        exec(f'wave{k}[:, ciclo] = ordinate2')
    j = 0
    # -- Onde test anni 18-19 --#
    # -- la  separazione è  dovuta alle differenti variabili per il set treaning e per il set test--#
    onda = {}
    for k in range(training + 1, test + 1, 1):
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
    mederr[r] = st.mean(maxerror[range(training, test, 1)])
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
    if t <= training:
        ascisse = data_tre.iloc[onda[0, 0]: onda[1, 0]]
        dati = "Treaning set"
    else:
        ascisse = data_t.iloc[onda[0, 0]: onda[1, 0]]
        dati = "Test set"
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
    stdrmse[t - 1] = 2 * val.std()
    REQM[t - 1] = rmse(ordinate1, ordinate2)  # (st.mean((ordinate1 - ordinate2) ** 2) ** 0.5)
    STDM[t - 1] = 2 * STD.mean()
    ErMAX[t - 1] = abs(maxonda1 - maxonda2)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    # xfmt = md.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(xfmt)
    line1 = plt.plot(datee, ordinate1, label='dati reali', ls='-', linewidth=1)
    line2 = plt.plot(datee, ordinate2, label='dati predetti', ls='-', linewidth=1)
    line3 = plt.plot(datee, (ordinate2 + STD), label='std sup', ls=':', mec='lime', linewidth=0.7)
    line4 = plt.plot(datee, (ordinate2 - STD), label='std inf', ls=':', mec='lime', linewidth=0.7)
    plt.annotate(dati + "  " + str(ascisse[0]) + '/ ' + str(ascisse[len(ascisse) - 1])+' Tolta piena 14', xy=(0.1, 0),
                 xycoords='axes fraction',
                 fontsize=7,
                 xytext=(200, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.annotate('onda n: ' + str(t), xy=(1, 0), xycoords='axes fraction',
                 fontsize=8,
                 xytext=(0, -25), textcoords='offset points',
                 ha='right', va='top')
    plt.legend(loc="upper right")
    plt.grid()
    plt.ylabel('[m] da zero idrometrico')
    ax = plt.gca()
    table_data = [
        ["Finestra(h):" + str(temp) + " Batch:" + str(batch) + "_Neu:" + str(neu), " Media [m]", "2 std [m]",
         "rmse [m]", "errore picchi [m]"],
         ["Singoli RMSE", np.around(medrmse[t - 1], 3), np.around(stdrmse[t - 1], 3), "/", "/"],
        ["Media valori", "/", np.around(STDM[t - 1], 3), np.around(REQM[t - 1], 3), np.around(ErMAX[t - 1], 3)],
    ]
    table = ax.table(cellText=table_data, loc='bottom', bbox=[0.0, -0.45, 1, .28])
    plt.subplots_adjust(bottom=0.2)
    table.set_fontsize(9)
    table.scale(1, 1)
    plt.savefig("Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + '_Batch' + str(
        batch) + '_Neu' + str(neu) + 'tagli14.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    # -- Save onda in txt mode --#
    ondaa = "Onda_Col" + str(t) + "_prediz(h)" + str(shift) + "Finestra(h)" + str(temp) + "_Batch" + str(
        batch) + "_Neu" + str(neu) + "tagli14.txt"
    with open(ondaa, "w") as out_file:
        for i in range(len(ascisse)):
            out_string = ""
            out_string += str(ascisse[i])
            out_string += "," + str(ordinate1[i])  # --ordinate2 è il dato reale
            out_string += "," + str(ordinate2[i])  # --ordinate2 è la predizione
            out_string += "," + str(ordinate2[i] + STD[i])
            out_string += "," + str(ordinate2[i] - STD[i])
            out_string += "\n"
            out_file.write(out_string)
print('END, SAVE ALL RMSE')  # -- fine dei cicli for--#
end = time.time()
t = str((end - start) / 60)
print('tempo:' + t + ' [min] _end prediction model')  # -- stampa a schermo il tempo impiegato--#
dat = "Colorno3_Finestra(h)" + str(temp) + "_recap_Batch" + str(batch) + "_Neu" + str(
    neu) + "tagli14.txt"  # -- salvataggio in txt delle variabili riassuntivi dei cicli for--#
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
