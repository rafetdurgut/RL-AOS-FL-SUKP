# 3 bolum halinde tekrar incele.
# en etkin 5 tanesinin zamana gore degisimlerini cizdir.
# Pareto kuralini uygula.


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

from Problem import *
import copy as cp
from typing import Tuple
def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):
    matrix = confusion_matrix(actual_classes, predicted_classes)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True,  cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()


def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 
    for train_ndx, test_ndx in kfold.split(X):
        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)
        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))
        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)
    return actual_classes, predicted_classes, predicted_proba
eps = 0.3
alpha = 0.9
gama = 0.5
pNo = 1
max_run=1
w = 25
reward='extreme'
learning=-1
operator_size=4
prob = OneMax(1000)
# prob = SetUnionKnapsack('Data/SUKP',pNo)
pName = prob.ID

feature_names = ["t","psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr","op_no"]
file_path = f"results/feature_information-CLRL-{operator_size}-{reward}-{eps}-{w}-{alpha}-{gama}-{learning}-None-0-NoneType-{pName}.csv"
# columns = ["op_no","iteration","run","f0","f1","f2","f3","f4","f5"] 
df = pd.read_csv(file_path, header=None)
df.columns = feature_names
feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
df = df.drop(['t'],axis=1)

X =  df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#Basic info
print(df.groupby('op_no').count())


# # ## Correlation Heatmap
# features = df.iloc[:,:-1]
# features.columns = feature_names
# plt.figure(figsize=(16, 6))
# heatmap = sns.heatmap(features.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
# plt.show()

#Chi-square 
# from sklearn.feature_selection import chi2
# chi_scores = chi2(X,y)
# sns.barplot(feature_names, chi_scores[0])
# plt.show()

# #Feature Importance RF
clf = RandomForestClassifier()
clf.fit(X, y)
y_pred_test = clf.predict(X)
c = accuracy_score(y, y_pred_test)
plt.barh(feature_names, clf.feature_importances_)
# sns.barplot(feature_names, clf.feature_importances_)
x = clf.feature_importances_
indices = np.argsort(x)
plt.xlabel("RF Feature Importance")
plt.ylabel("Features")
plt.show()

#Classification SVC
# model = SVC(kernel='linear')
# model.fit(X, y)
# feature_importance=[abs(i) for i in model.coef_[0]]
# sns.barplot(feature_names, feature_importance)
# plt.show()


#Classification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
kfold = KFold(n_splits=5, random_state=42, shuffle=True)


RFC = RandomForestClassifier()
SVCL = SVC(kernel='linear')
MLP = MLPClassifier()

models = [RFC,SVCL, MLP]
for m in models:
    actual_classes, predicted_classes, _ = cross_val_predict(m, kfold, X, y)
    plot_confusion_matrix(actual_classes, predicted_classes, ["0", "1", "2","3"])
    print(accuracy_score(actual_classes, predicted_classes))


# RFC_results = cross_val_score(RFC, X, y, cv=kfold, scoring='accuracy', verbose=10)
# print(RFC_results.mean())

# SVCL_results = cross_val_score(SVCL, X, y, cv=kfold, scoring='accuracy', verbose=10)
# print(SVCL_results.mean())
 
# MLP_results = cross_val_score(MLP, X, y, cv=kfold, scoring='accuracy', verbose=10)
# print(MLP_results.mean())


# SVCL.fit(X_train, y_train)
# SVCL_y_pred_test = SVCL.predict(X_test)
# c = accuracy_score(y_test, SVCL_y_pred_test)
# print(c)

# MLP.fit(X_train, y_train)
# MLP_y_pred_test = MLP.predict(X_test)
# c = accuracy_score(y_test, MLP_y_pred_test)
# print(c)

# from sklearn.metrics import confusion_matrix
# cf_matrix = confusion_matrix(y_test, RFC_y_pred_test)
# sns.heatmap(cf_matrix, annot=True)
# plt.show()

# cf_matrix = confusion_matrix(y_test, SVCL_y_pred_test)
# sns.heatmap(cf_matrix, annot=True)
# plt.show()

# cf_matrix = confusion_matrix(y_test, MLP_y_pred_test)
# sns.heatmap(cf_matrix, annot=True)
# plt.show()

#Visualize Features
# x= np.arange(len(features["idb"]))
# sns.scatterplot(x,features["idb"].values)
# plt.show()