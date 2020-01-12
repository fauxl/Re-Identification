from numpy.distutils.system_info import numarray_info
from scipy.spatial import distance
import csv
import sys
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def plot_roc_curve(fpr, tpr):
    x =  fpr
    y =  tpr
    plt.plot(x, y, color='orange', linestyle='solid' )
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='solid')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


pin = [ 'Workclass', 'Education', 'Occupation',  'Race', 'Native-Country', ]

features_per_allenamento = pd.read_csv (r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\training.csv', usecols = ['Age', 'Workclass', 'Education',
                                                                                                                                                     'Marital-status', 'Occupation', 'Relationship',
                                                                                                                                                     'Race', 'Sex', 'Native-Country',
                                                                                                                                                    'Income'])
features_per_predire = pd.read_csv (r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\adult.csv', usecols = ['Age', 'Workclass', 'Education',
                                                                                                                                                                'Marital-status', 'Occupation', 'Relationship',
                                                                                                                                                                 'Race', 'Sex', 'Native-Country',
                                                                                                                                                                 'Income'])
features_per_encoder = features_per_predire

# fitto il LabelEncoder con tutte le features usando adult.csv che contiene tutte le possibili classi per ogni feature
label_encoder = preprocessing.LabelEncoder()
dataframe_predizione = pd.DataFrame()
dataframe_training = pd.DataFrame()


## Features Selection Array

#features_per_allenamento = features_per_allenamento.drop(columns=pin[0:5])
#features_per_predire = features_per_predire.drop(columns=pin[0:5])
#features_per_encoder = features_per_encoder.drop(columns=pin[0:5])

print(features_per_predire)
# For da 0 a 9 per trasformare gli array di features da Stringhe ad numeri Float
for x in range(0,10):
    singola_feature_predizione = features_per_predire.iloc[:, x]
    singola_feature_training = features_per_allenamento.iloc[:, x]
    singola_feature_encoder = features_per_encoder.iloc[:, x]

    #print(list(features_per_allenamento.iloc[:, x])) #stampo le features utilizzate per l identificazione
    label_encoder.fit(singola_feature_encoder)  # fitto l'encoder con tutte le possibili classi della feature

    # presa una colonna di feature, normalizzo i dati della feature del dataset adult.csv
    singola_feature_predizione = pd.DataFrame(label_encoder.transform(singola_feature_predizione))
    dataframe_predizione = pd.concat([dataframe_predizione, singola_feature_predizione], axis=1)

    label_encoder.fit(singola_feature_encoder)  # fitto l'encoder con tutte le possibili classi della feature, DI NUOVO

    # per la stessa feature, normalizzo i dati del dataset di training.csv
    singola_feature_training = pd.DataFrame(label_encoder.transform(singola_feature_training))
    dataframe_training = pd.concat([dataframe_training, singola_feature_training], axis=1)

classi_target = pd.read_csv(r'C:\\Users\\FauxL\\Desktop\\Data Science\\Project\\training.csv', usecols=['Name'])
classificatore = RandomForestClassifier(n_estimators = 150, criterion='entropy', n_jobs=4)
classificatore.fit(dataframe_training, classi_target.values.ravel()) # fitto con tutto il dataset di training

# testo il classificatore con delle classi che già conosce e di cui conosce anche l array di feature per testarne la precisione
predict = classificatore.predict(dataframe_training)
predict_proba = classificatore.predict_proba(dataframe_training)
predict = np.array(predict)
classi_target = list(np.array(classi_target))

cnf_matrix = confusion_matrix(classi_target, predict)
print(' - Confusion Matrix -')
print(cnf_matrix)
print(' - Accuracy Score -', accuracy_score(classi_target, predict))
print(' - Report  -'), print(classification_report(classi_target, predict))

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

# Utilizzo la funzione plot_roc_curve() definita sopra per disegnare il grafico della roc curve
skplt.metrics.plot_roc(classi_target, predict_proba)
plt.show()
