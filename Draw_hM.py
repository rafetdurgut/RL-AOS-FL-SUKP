# 3 bolum halinde tekrar incele.
# en etkin 5 tanesinin zamana gore degisimlerini cizdir.
# Pareto kuralini uygula.

#%%
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
def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list,fileName):
    matrix = confusion_matrix(actual_classes, predicted_classes)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True,  cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(fileName)
    plt.savefig(f"figs/{fileName}.png")
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
eps = 0.5
alpha = 0.5
gama = 0.3
pNo = 1
max_run=1
w = 25
reward='extreme'
learning=-1
operator_size=4
max_iteration = 500

# prob = OneMax(1000)
prob = SetUnionKnapsack('Data/SUKP',pNo)
pName = prob.ID

feature_names = ["t","psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr","op_no"]
file_path = f"results/feature_information-CLRL-{operator_size}-{reward}-{eps}-{w}-{alpha}-{gama}-{learning}-{pName}.csv"
# columns = ["op_no","iteration","run","f0","f1","f2","f3","f4","f5"] 
df = pd.read_csv(file_path, header=None,names=feature_names)
# df.columns = feature_names
# # feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
temp_data = df[df["t"]<(max_iteration/3)]
temp_data2 = df[(df["t"]>(max_iteration/3) ) & (df["t"]<2*(max_iteration/3))]
temp_data3 = df[df["t"]>2*(max_iteration/3)]
acc = []
def perform_analysis(data, title):
    #part 1
    org_data = deepcopy(data)
    data = data.drop(['t'],axis=1)
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    #Basic info
    print(data.groupby('op_no').count())
    feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
    # # ## Correlation Heatmap
    features = data.iloc[:,:-1]
    features.columns = feature_names
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(features.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title(f"Correlation Heatmap - Part {title}", fontdict={'fontsize':18}, pad=12);
    plt.savefig(f"figs/HM-{title}.png")
    plt.show()
    plt.close()

    #Chi-square 
    from sklearn.feature_selection import chi2
    chi_scores = chi2(X,y)
    plt.figure(figsize=(16, 6))

    sns.barplot(feature_names, chi_scores[0]).set_title(f"Chi-square - Feature Importances - Section {title}")
    plt.xlabel("Coefficients")
    plt.ylabel("Features")
    plt.savefig(f"figs/CH-{title}.png")
    plt.show()
    plt.close()

    # #Feature Importance RF
    clf = RandomForestClassifier()
    clf.fit(X, y)
    y_pred_test = clf.predict(X)
    c = accuracy_score(y, y_pred_test)
    # plt.barh(feature_names, clf.feature_importances_)
    plt.figure(figsize=(16, 6))
    sns.barplot(feature_names, clf.feature_importances_).set_title(f"RF - Feature Importance - Section {title}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.savefig(f"figs/RF-FI-{title}.png")
    plt.show()
    plt.close()

    #Classification SVC
    model = SVC(kernel='linear')
    model.fit(X, y)
    feature_importance=[abs(i) for i in model.coef_[0]]
    plt.figure(figsize=(16, 6))
    sns.barplot(feature_names, feature_importance).set_title(f"SVM Feature Coefficients - Section {title}")
    plt.xlabel("Coefficients")
    plt.ylabel("Features")
    plt.savefig(f"figs/SVC-FI-{title}.png")
    plt.show()
    plt.close()



    #Classification
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    kfold = KFold(n_splits=5, random_state=0, shuffle=True)


    RFC = RandomForestClassifier()
    SVCL = SVC(kernel='linear')
    MLP = MLPClassifier()
    titles =[f"RF - Section {title}",f"SVM - Section {title}",f"MLP - Section {title}"]
    models = [RFC,SVCL, MLP]
    for ind,m in enumerate(models):
        actual_classes, predicted_classes, _ = cross_val_predict(m, kfold, X, y)
        plot_confusion_matrix(actual_classes, predicted_classes, ["0", "1", "2","3"],titles[ind])
        acc.append(accuracy_score(actual_classes, predicted_classes))




    # Visualize Features
    #Box plot olarak cizdir.
    #5 tanesi
    # org_data =  org_data.astype({'t': int})

    # sns.boxplot(x="t",y="idb", data=org_data).set_title(f'Feature idb - Section {title}')
    # plt.xlabel('iteration')
    # plt.ylabel('value')
    # plt.savefig(f"figs/feature-idb-{title}.png")
    # plt.show()

    # sns.lineplot(x="t",y="osr", hue="op_no",data=org_data).set_title(f'Feature osr - Section {title}')
    # plt.xlabel('iteration')
    # plt.ylabel('value')
    # plt.savefig(f"figs/feature-osr-{title}.png")
    # plt.show()
# plt.figure(figsize=(16, 6))

# sns.set_theme(palette="tab10")
# sns.lineplot(x="t",y="osr", hue="op_no",data=df,palette="tab10").set_title(f'Feature osr - ')
# plt.xlabel('iteration')
# plt.ylabel('value')
# plt.xticks(np.arange(0,max_iteration,10))
# plt.savefig(f"figs/feature-osr.png")
# plt.show()
# plt.close()
# plt.figure(figsize=(16, 6))

# sns.boxplot(x="t",y="idb", data=df,palette="tab10").set_title(f'Feature idb')
# plt.xlabel('iteration')
# plt.ylabel('value')
# plt.xticks(np.arange(0,max_iteration,10))

# plt.savefig(f"figs/feature-idb.png")
# plt.show()
# plt.close()

# perform_analysis(temp_data,1)
# perform_analysis(temp_data2,2)
# perform_analysis(temp_data3,3)


temp_data = temp_data.drop(['t'],axis=1)
X = temp_data.iloc[:,:-1].values
y = temp_data.iloc[:,-1].values
#Basic info
print(temp_data.groupby('op_no').count())
feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
# # ## Correlation Heatmap
features = temp_data.iloc[:,:-1]
features.columns = feature_names

clf = RandomForestClassifier()
fig, axes = plt.subplots(3, 1,figsize=(16, 6))
clf.fit(X, y)



heatmap = sns.heatmap(features.corr(), ax=axes[0], vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title(f"Correlation Heat Map", fontdict={'fontsize':18}, pad=12);

temp_data2 = temp_data2.drop(['t'],axis=1)
X = temp_data2.iloc[:,:-1].values
y = temp_data2.iloc[:,-1].values
#Basic info
print(temp_data2.groupby('op_no').count())
feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
# # ## Correlation Heatmap
features = temp_data2.iloc[:,:-1]
features.columns = feature_names

clf2 = RandomForestClassifier()
clf2.fit(X, y)

heatmap = sns.heatmap(features.corr(),  ax=axes[1], vmin=-1, vmax=1, annot=False, cmap='BrBG')

temp_data3 = temp_data3.drop(['t'],axis=1)
X = temp_data3.iloc[:,:-1].values
y = temp_data3.iloc[:,-1].values
#Basic info
print(temp_data3.groupby('op_no').count())
feature_names = ["psd","pfd","pic","pnb","pai","pcv","pcr","eap","evp","atn","pdd","idg","idp","ifg","ifp","idb","idw","itn","osr"]
# # ## Correlation Heatmap
features = temp_data3.iloc[:,:-1]
features.columns = feature_names

clf3 = RandomForestClassifier()
clf3.fit(X, y)

heatmap = sns.heatmap(features.corr(), ax=axes[2],  vmin=-1, vmax=1, annot=False, cmap='BrBG')
plt.xlabel("Features")
plt.savefig(f"figs/HM-FI-SUKP.pdf")
plt.show()
plt.close()





print(acc)
# %%
