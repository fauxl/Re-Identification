import csv
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier



COLUMNS = ['Age', 'Workclass', 'Education',    'Marital-status', 'Occupation', 'Relationship',  'Race', 'Sex', 'Native-Country', 'Income']
pin = [ 'Workclass', 'Education', 'Occupation',  'Race', 'Native-Country', ]

features_per_allenamento = pd.read_csv (r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\training.csv', usecols = ['Age', 'Workclass', 'Education',
                                                                                                                                                     'Marital-status', 'Occupation', 'Relationship',
                                                                                                                                                     'Race', 'Sex', 'Native-Country', 'Income'])
features_per_predire = pd.read_csv (r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\adult.csv', usecols = ['Age', 'Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship',
                                                                                                                                                                 'Race', 'Sex', 'Native-Country', 'Income'])
features_per_encoder = features_per_predire

# fitto il LabelEncoder con tutte le features usando adult.csv che contiene tutte le possibili classi per ogni feature
label_encoder = preprocessing.LabelEncoder()
dataframe_predizione = pd.DataFrame()
dataframe_training = pd.DataFrame()

## Features Selection Array

features_per_allenamento = features_per_allenamento.drop(columns=pin[0:5])
features_per_predire = features_per_predire.drop(columns=pin[0:5])
features_per_encoder = features_per_encoder.drop(columns=pin[0:5])


# For da 0 a 9 per trasformare gli array di features da Stringhe ad numeri Float
for x in range(0,5):
    singola_feature_predizione = features_per_predire.iloc[:, x]
    singola_feature_training = features_per_allenamento.iloc[:, x]
    singola_feature_encoder = features_per_encoder.iloc[:, x]

    label_encoder.fit(singola_feature_encoder)  # fitto l'encoder con tutte le possibili classi della feature

    # presa una colonna di feature, normalizzo i dati della feature del dataset adult.csv
    singola_feature_predizione = pd.DataFrame(label_encoder.transform(singola_feature_predizione))
    dataframe_predizione = pd.concat([dataframe_predizione, singola_feature_predizione], axis=1)

    label_encoder.fit(singola_feature_encoder)  # fitto l'encoder con tutte le possibili classi della feature, DI NUOVO

    # per la stessa feature, normalizzo i dati del dataset di training.csv
    singola_feature_training = pd.DataFrame(label_encoder.transform(singola_feature_training))
    dataframe_training = pd.concat([dataframe_training, singola_feature_training], axis=1)

classi_target = pd.read_csv(r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\training.csv', usecols=['Name'])
classificatore = RandomForestClassifier(n_estimators = 250, criterion='entropy', n_jobs=4)
classificatore.fit(dataframe_training, classi_target.values.ravel()) # fitto con tutto il dataset di training
print("\nRilevanza attributi")
for name, score in zip(COLUMNS, classificatore.feature_importances_):
    print(name, score)
# testo il classificatore con delle classi che già conosce e di cui conosce anche l array di feature per testarne la precisione
predict = classificatore.predict(dataframe_training)
predict = np.array(predict)
classi_target = list(np.array(classi_target))

# Predico utilizzando il dataset dell'esperimento
predizione = classificatore.predict(dataframe_predizione)

#calcolo per ogni risultato la distanza euclidea dall'array predetto
dataframe_risultati = pd.DataFrame(predizione)

# prelevo gli array obiettivo dal dataset training.csv e li assegno

array_name=  pd.read_csv (r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\training.csv', usecols = ["Name"]).to_numpy()
#print(array_name)

risultati_con_distanza = pd.DataFrame()
arrayDistanze = [ ]

# For per il calcolo delle distanze euclidee
somma_tot_distanze = 0
for y in range(1, dataframe_risultati.size):
    nome_predetto = dataframe_risultati.iloc[y].values # prendo l identità predetta dell' y-esimo elemento
    array_features = dataframe_predizione.iloc[y] # prendo l' array di features dell' y-esimo elemento
    j=0
    while j < len(array_name):
        if nome_predetto == array_name[j]:
            array_obiettivo = dataframe_training.iloc[j]
            break
        j=j+1

    dst = distance.euclidean(array_features, array_obiettivo)
    nome_predetto = pd.DataFrame(dataframe_risultati.iloc[y])

    somma_tot_distanze = somma_tot_distanze + dst
    # print('elemento n [' , y, '] con identità ' , nome_predetto.values, ' con distanza: [' , dst , '] da array obiettivo: ', array_obiettivo.values  )
    arrayDistanze.append(dst)

# scrivo i risultati predetti in un file csv
with open('predizioni.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter=' ')
    writer.writerows(predizione)
csvFile.close()

#scrivo le distanze calcolate del rispettivo y-esimo elemento dall' identità data dall'estimatore
with open('distanza.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter=' ')
    writer.writerows(map(lambda x: [x], arrayDistanze))
csvFile.close()

# conto il numero di predizione per ogni classe
np.set_printoptions(threshold = sys.maxsize)
unici, counteggio = np.unique(dataframe_risultati, return_counts=True)
print(unici,counteggio)

# Grafico a torta
# plt.pie(counteggio,  labels=unici, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
#plt.show()

# do in output la distanza media come indice di errore per testare
print('valore medio delle distanze: ', somma_tot_distanze/dataframe_risultati.size)