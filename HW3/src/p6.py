'''
Created on Nov 26, 2017

@author: Amir
'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import product
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
from p1 import getInput


import warnings
warnings.filterwarnings("ignore")


x, _, y = getInput()
xPCA = np.copy(x)

pca = PCA(30)
x = pca.fit_transform(x)

pca = PCA(3)
xPCA = pca.fit_transform(xPCA)

print(x.shape, xPCA.shape)

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
# print(y[1,:])
numClasses = y.shape[1]

numSplits = 10
yHat = np.empty((0,96))
tHat = np.empty((0,96))
tPCAHat = np.empty((0,96))

kf = KFold(n_splits=numSplits, shuffle = True, random_state = 69)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    xPCA_train, xPCA_test = xPCA[train_index], xPCA[test_index]
    y_train, y_test = y[train_index], y[test_index]
    T = compositeClassifier.fit(x_train, y_train).decision_function(x_test)
    TPCA = compositeClassifier.fit(xPCA_train, y_train).decision_function(xPCA_test)
    yHat = np.append(yHat, y_test, axis=0)
    tHat = np.append(tHat, T, axis=0)
    tPCAHat = np.append(tPCAHat, TPCA, axis=0)

fpr,tpr, _ = roc_curve(yHat.ravel(), tHat.ravel())
roc_auc = roc_auc_score(yHat.ravel(), tHat.ravel())

precision, recall, _ = precision_recall_curve(yHat.ravel(), tHat.ravel())
pr_auc = average_precision_score(yHat, tHat, average="micro")

# for problem 6
fprPCA,tprPCA, _ = roc_curve(yHat.ravel(), tPCAHat.ravel())
roc_aucPCA = roc_auc_score(yHat.ravel(), tPCAHat.ravel())

precisionPCA, recallPCA, _ = precision_recall_curve(yHat.ravel(), tPCAHat.ravel())
pr_aucPCA = average_precision_score(yHat, tPCAHat, average="micro")





#plotting
fig = plt.figure()
axLeft = fig.add_subplot(121)
axRight = fig.add_subplot(122)
axLeft.plot(fpr, tpr, label = 'PC:30 (area = %0.2f)' % roc_auc)
axLeft.plot(fprPCA, tprPCA, label = 'PC:3 (area = %0.2f)' % roc_aucPCA)
axLeft.plot([0, 1], [0, 1], color='navy', linestyle='--')

axRight.plot(precision, recall, label = 'PC:30 (area = %0.2f)' % pr_auc)
axRight.plot(precisionPCA, recallPCA, label = 'PC:3 (area = %0.2f)' % pr_aucPCA)

axLeft.legend(bbox_to_anchor=(-.08, 1), fontsize = 12)
axRight.legend(bbox_to_anchor=(1.1, 1), fontsize = 12)

axLeft.set_title("Micro ROC Curve")
axLeft.set_xlabel('False Positive Rate')
axLeft.set_ylabel('True Positive Rate')

axRight.set_title("Micro PR Curve")
axRight.set_xlabel('Precision')
axRight.set_ylabel('Recall')

fig.suptitle("Problem 6: Composite SVM Classifier Performance (PC 30 vs PC 3)")

plt.show()
