from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

""""
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(dataframe_training,classi_target.values.ravel())

y2 = classi_target.values.ravel()- tree_reg1.predict(dataframe_training)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(dataframe_training, y2)

y3 = y2 - tree_reg2.predict(dataframe_training)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(dataframe_training, y3)

y_pred = sum(tree.predict(dataframe_predizione) for tree in (tree_reg1, tree_reg2, tree_reg3))

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(dataframe_training, classi_target.values.ravel())

"""

# testo il classificatore con delle classi che già conosce e di cui conosce anche l array di feature per testarne la precisione
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(dataframe_training, classi_target.values.ravel())

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

plt.show()

