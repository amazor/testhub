'''
Created on Nov 26, 2017

@author: Amir
'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import product
import numpy as np
from sklearn.metrics.ranking import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

from p1 import getInput

x, _, y = getInput()

pca = PCA(30)
x = pca.fit_transform(x)

yEnv_Pert = y[:,3]
yGene_Pert = y[:,4]
yList =(list(zip(yEnv_Pert, yGene_Pert)))
yList = ["".join(tuple) for tuple in yList]

envClass= ["Indole", "O2-starvation", "RP-overexpress", "Antibacterial", "Carbon-limitation",
            "Dna-damage", "Zinc-limitation", "none"]
geneClass= ["appY_KO","arcA_KO","argR_KO", "cya_KO", "fis_OE", "fnr_KO", "frdC_KO", "na_WT", "oxyR_KO",
             "rpoS_KO", "soxS_KO", "tnaA_KO"]
classList = (list(product(envClass, geneClass)))
classList = ["".join(tuple) for tuple in classList]

# print(yList[1])
# print(classList)
compositeClassifier = OneVsRestClassifier(svm.SVC(C= 5))
y = label_binarize(yList, classes=classList)
yEnv = label_binarize(yEnv_Pert, classes=envClass)
yGene = label_binarize(yGene_Pert, classes=geneClass)
# print(y[1,:])
numClasses = y.shape[1]
numEnvClasses = yEnv.shape[1] 
numGeneClasses = yGene.shape[1] 

numSplits = 10



yHat = np.empty((0,numClasses))
tHat = np.empty((0,numClasses))
yEnvHat = np.empty((0,numEnvClasses))
tEnvHat = np.empty((0,numEnvClasses))
yGeneHat = np.empty((0,numGeneClasses))
tGeneHat = np.empty((0,numGeneClasses))

kf = KFold(n_splits=numSplits, shuffle = True, random_state = 69)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    yEnv_train, yEnv_test = yEnv[train_index], yEnv[test_index]
    yGene_train, yGene_test = yGene[train_index], yGene[test_index]
    T = compositeClassifier.fit(x_train, y_train).decision_function(x_test)
    TEnv = compositeClassifier.fit(x_train, yEnv_train).decision_function(x_test)
    TGene = compositeClassifier.fit(x_train, yGene_train).decision_function(x_test)  
      
    yHat = np.append(yHat, y_test, axis=0)
    tHat = np.append(tHat, T, axis=0)
    yEnvHat = np.append(yEnvHat, yEnv_test, axis=0)
    tEnvHat = np.append(tEnvHat, TEnv, axis=0)
    yGeneHat = np.append(yGeneHat, yGene_test, axis=0)
    tGeneHat = np.append(tGeneHat, TGene, axis=0)


fpr,tpr, _ = roc_curve(yHat.ravel(), tHat.ravel())
roc_auc = roc_auc_score(yHat.ravel(), tHat.ravel())

precision, recall, _ = precision_recall_curve(yHat.ravel(), tHat.ravel())
pr_auc = average_precision_score(yHat, tHat, average="micro")

fprEnv,tprEnv, _ = roc_curve(yEnvHat.ravel(), tEnvHat.ravel())
rocEnv_auc = roc_auc_score(yEnvHat.ravel(), tEnvHat.ravel())

precisionEnv, recallEnv, _ = precision_recall_curve(yEnvHat.ravel(), tEnvHat.ravel())
prEnv_auc = average_precision_score(yEnvHat, tEnvHat, average="micro")

fprGene,tprGene, _ = roc_curve(yGeneHat.ravel(), tGeneHat.ravel())
rocGene_auc = roc_auc_score(yGeneHat.ravel(), tGeneHat.ravel())

precisionGene, recallGene, _ = precision_recall_curve(yGeneHat.ravel(), tGeneHat.ravel())
prGene_auc = average_precision_score(yGeneHat, tGeneHat, average="micro")

fig = plt.figure()
axLeft = fig.add_subplot(121)
axRight = fig.add_subplot(122)
axLeft.plot(fpr, tpr, label = 'Composite (area = %0.2f)' % roc_auc)
axLeft.plot(fprEnv, tprEnv, label = 'Env_Pert (area = %0.2f)' % rocEnv_auc)
axLeft.plot(fprGene, tprGene, label = 'Gene_Pert (area = %0.2f)' % rocGene_auc)
axLeft.plot([0, 1], [0, 1], color='navy', linestyle='--')

axRight.plot(precision, recall, label = 'Composite (area = %0.2f)' % pr_auc)
axRight.plot(precisionEnv, recallEnv, label = 'Env_Pert (area = %0.2f)' % prEnv_auc)
axRight.plot(precisionGene, recallGene, label = 'Gene_Pert (area = %0.2f)' % prGene_auc)


axLeft.legend(bbox_to_anchor=(-.02, 1), fontsize = 9)
axRight.legend(bbox_to_anchor=(1.1, 1), fontsize = 9)

axLeft.set_title("Micro ROC Curve")
axLeft.set_xlabel('False Positive Rate')
axLeft.set_ylabel('True Positive Rate')

axRight.set_title("Micro PR Curve")
axRight.set_xlabel('Precision')
axRight.set_ylabel('Recall')

fig.suptitle("Problem 5: Composite SVM Classifier Performance")

plt.show()
